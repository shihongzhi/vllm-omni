# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
import threading
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

# Environment switch for the cross-process put notification (see
# ``SharedMemoryConnector`` below).  Off by default: the notification is a
# latency hint for the receiver's recv loop, and deployments that prefer the
# established poll behaviour keep it unchanged.
_CHUNK_NOTIFY_ENV = "VLLM_OMNI_CHUNK_NOTIFY"
# Optional suffix separating concurrent deployments' notification sockets on
# the same host (tests and multi-instance setups).  A collision is not a
# correctness issue -- the receiver's bind fails and its recv loop keeps
# polling -- but a distinct salt keeps the hint useful.
_CHUNK_NOTIFY_SALT_ENV = "VLLM_OMNI_CHUNK_NOTIFY_SALT"


def _chunk_notify_env_enabled() -> bool:
    return os.environ.get(_CHUNK_NOTIFY_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _notify_endpoint(from_stage: str, to_stage: str) -> str:
    salt = os.environ.get(_CHUNK_NOTIFY_SALT_ENV, "").strip()
    suffix = f"_{salt}" if salt else ""
    return f"ipc:///tmp/vllm_omni_chunk_notify_{from_stage}_{to_stage}{suffix}"


class SharedMemoryConnector(OmniConnectorBase):
    """Key-addressed local shared-memory connector.

    SHM is a local-only transport: it reads/writes POSIX shared memory
    segments identified purely by *key*.  It does **not** understand
    remote-transport metadata such as ``source_host`` / ``source_port``
    (that is the RDMA connector's job).  When such metadata is passed in,
    the connector silently falls back to key-based lookup.

    Optionally (``extra["chunk_notify"]`` or ``VLLM_OMNI_CHUNK_NOTIFY=1``)
    the connector pairs each SHM edge with a local ZMQ PUSH/PULL socket
    keyed by the same ``(from_stage, to_stage)`` pair the ``put``/``get``
    calls already carry.  A successful ``put`` pushes one zero-payload
    hint so the receiver's recv thread can sleep on the socket instead of
    polling ``shm_open`` on a fixed backoff.  The data plane is untouched:
    a dropped or lost hint only costs the receiver its fallback poll.
    """

    supports_chunk_notify = True

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self._pending_keys: set[str] = set()
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
        }

        # Per-edge opt-in (deploy yaml ``extra["chunk_notify"]``) wins over
        # the global environment switch so a single deployment can A/B the
        # notification channel edge by edge.
        explicit_notify = config.get("chunk_notify")
        self._chunk_notify_enabled = bool(explicit_notify) if explicit_notify is not None else _chunk_notify_env_enabled()

        # Notification sockets are strictly thread-affine: PUSH is only
        # created on the sender (save-loop) thread, PULL only on the
        # receiver (recv-loop) thread.  ZeroMQ sockets must not be shared
        # across threads, so the caches remember their owning thread and
        # rebuild on a foreign thread instead of reusing.
        self._notify_ctx = None
        self._push_sockets: dict[tuple[str, str], Any] = {}
        self._pull_sockets: dict[tuple[str, str], Any] = {}
        self._pull_bind_failed: set[tuple[str, str]] = set()
        self._notify_lock = threading.Lock()
        self._notify_dropped = 0

    # --- Optional chunk-notification protocol ---

    def chunk_notify_enabled(self) -> bool:
        return self._chunk_notify_enabled

    def _notify_context(self):
        # A dedicated context (rather than a global singleton) keeps close()
        # able to tear everything down without affecting other connectors.
        if self._notify_ctx is None:
            import zmq

            self._notify_ctx = zmq.Context()
        return self._notify_ctx

    def _get_push_socket(self, from_stage: str, to_stage: str):
        import zmq

        edge = (from_stage, to_stage)
        entry = self._push_sockets.get(edge)
        if entry is not None and entry[0] == threading.get_ident():
            return entry[1]
        if entry is not None:
            self._close_socket(entry[1])
        sock = self._notify_context().socket(zmq.PUSH)
        # The receiver binds; the sender connects.  A PUSH that has
        # connected but not yet reached the receiver queues hints in its
        # local pipe (up to SNDHWM), so a hint fired before the receiver
        # starts is delayed, not lost -- unlike a bound PUSH, whose
        # non-blocking send with no peer drops immediately.
        sock.setsockopt(zmq.SNDHWM, 1024)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(_notify_endpoint(from_stage, to_stage))
        self._push_sockets[edge] = (threading.get_ident(), sock)
        return sock

    def _get_pull_socket(self, from_stage: str, to_stage: str):
        import zmq

        edge = (from_stage, to_stage)
        if edge in self._pull_bind_failed:
            return None
        entry = self._pull_sockets.get(edge)
        if entry is not None and entry[0] == threading.get_ident():
            return entry[1]
        if entry is not None:
            self._close_socket(entry[1])
        sock = self._notify_context().socket(zmq.PULL)
        sock.setsockopt(zmq.RCVHWM, 1024)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.bind(_notify_endpoint(from_stage, to_stage))
        except Exception as e:
            # Another receiver already bound this endpoint (same host,
            # same edge numbering).  The hint is best-effort; keep the
            # poll behaviour instead of failing the stage.  Cache the
            # failure so the per-poll get() does not retry the bind (and
            # churn a socket) on every pass.
            sock.close()
            self._pull_bind_failed.add(edge)
            logger.debug("chunk-notify bind failed for edge %s->%s: %s", from_stage, to_stage, e)
            return None
        self._pull_sockets[edge] = (threading.get_ident(), sock)
        return sock

    @staticmethod
    def _close_socket(sock) -> None:
        try:
            sock.close()
        except Exception:
            pass

    def notify_chunk_put(self, from_stage: str, to_stage: str, put_key: str) -> None:
        if not self._chunk_notify_enabled:
            return
        try:
            import zmq

            sock = self._get_push_socket(from_stage, to_stage)
            if sock is None:
                return
            try:
                sock.send(put_key.encode(), flags=zmq.DONTWAIT)
            except zmq.Again:
                self._notify_dropped += 1
        except Exception as e:
            # The notification is a pure hint on the send path: any failure
            # here must be invisible to the chunk that was already written.
            logger.debug("chunk-notify send failed for %s: %s", put_key, e)

    def wait_for_chunk_notify(self, timeout_ms: int, wake_fds: tuple[int, ...] = ()) -> None:
        import time

        if not self._chunk_notify_enabled or (not self._pull_sockets and not wake_fds):
            # Disabled, or nothing to sleep on yet (no receive edge was
            # ever polled): a plain sleep keeps the recv loop's cadence
            # instead of degenerating into a busy spin.
            time.sleep(max(0, timeout_ms) / 1000.0)
            return
        try:
            import zmq

            poller = zmq.Poller()
            for entry in self._pull_sockets.values():
                poller.register(entry[1], zmq.POLLIN)
            for fd in wake_fds:
                poller.register(fd, zmq.POLLIN)
            poller.poll(timeout=max(0, timeout_ms))
            # Drain whatever arrived so the next wait starts empty.  The
            # payload (the put key) is intentionally unread: the recv loop
            # re-scans its whole pending set, so a hint only needs to wake
            # it, not tell it which key landed.
            for entry in list(self._pull_sockets.values()):
                try:
                    while True:
                        entry[1].recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
        except Exception as e:
            logger.debug("chunk-notify wait failed: %s", e)
            time.sleep(max(0, timeout_ms) / 1000.0)

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            payload = self.serialize_obj(data)
            size = len(payload)

            lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                meta = shm_write_bytes(payload, name=put_key)
                fcntl.flock(lockf, fcntl.LOCK_UN)

            if self._chunk_notify_enabled:
                self.notify_chunk_put(from_stage, to_stage, put_key)

            # meta contains {'name': ..., 'size': ...}
            metadata = {"shm": meta, "size": size}
            self._pending_keys.add(put_key)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict[str, Any]) -> tuple[Any, int] | None:
        deserialized = False
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                data_bytes = shm_read_bytes(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            obj = self.deserialize_obj(data_bytes)
            result = (obj, int(shm_handle.get("size", 0)))
            deserialized = True
            return result
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            if deserialized:
                try:
                    os.remove(lock_file)
                except FileNotFoundError:
                    pass

    def _get_by_key(self, get_key: str) -> tuple[Any, int] | None:
        """Read a SHM segment addressed purely by *get_key*."""
        shm = None
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            if shm is None or shm.size == 0:
                return None
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            shm_handle = {"name": get_key, "size": shm.size}
            result = self._get_data_with_lock(lock_file, shm_handle)
            if result is not None:
                self._pending_keys.discard(get_key)
            return result
        except FileNotFoundError:
            return None
        except ValueError as e:
            # A receiver can observe a newly-created POSIX SHM object before
            # the writer has finished sizing it. Treat that as "not ready yet"
            # so async polling can retry without a traceback.
            if "empty file" in str(e):
                return None
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        except Exception:
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        finally:
            if shm:
                shm.close()

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        # The first poll on an edge registers it as a receive edge: the
        # recv thread's PULL socket must exist before wait_for_chunk_notify
        # can sleep on it.  Misses (chunk not yet written) still return
        # None below; registering here keeps the notification channel
        # ahead of the first possible hit.
        if self._chunk_notify_enabled:
            try:
                self._get_pull_socket(from_stage, to_stage)
            except Exception as e:
                logger.debug("chunk-notify pull setup failed for edge %s->%s: %s", from_stage, to_stage, e)
        if metadata is not None:
            if isinstance(metadata, dict) and get_key in metadata:
                metadata = metadata.get(get_key)

            if isinstance(metadata, dict) and "shm" in metadata:
                shm_handle = metadata["shm"]
                lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                result = self._get_data_with_lock(lock_file, shm_handle)
                if result is not None:
                    self._pending_keys.discard(get_key)
            else:
                # Missing or non-SHM metadata falls back to key-based lookup.
                result = self._get_by_key(get_key)
        else:
            result = self._get_by_key(get_key)

        if result is not None:
            self._metrics["gets"] += 1
        return result

    def cleanup(self, request_id: str) -> None:
        """Best-effort cleanup of unconsumed SHM segments for *request_id*.

        Matches pending keys where *request_id* appears as the full key,
        as a ``_``-delimited prefix, or as a ``_``-delimited suffix.
        If ``get()`` was never called, we unlink it here so /dev/shm
        doesn't leak.
        """
        stale = [
            k
            for k in self._pending_keys
            if k == request_id or k.startswith(request_id + "_") or k.endswith("_" + request_id)
        ]
        for key in stale:
            self._pending_keys.discard(key)
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
                logger.debug("cleanup: unlinked unconsumed SHM segment %s", key)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("cleanup: failed to unlink SHM segment %s: %s", key, e)
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

    def close(self) -> None:
        """Unlink all remaining tracked SHM segments."""
        with self._notify_lock:
            for entry in list(self._push_sockets.values()):
                self._close_socket(entry[1])
            self._push_sockets.clear()
            for entry in list(self._pull_sockets.values()):
                self._close_socket(entry[1])
            self._pull_sockets.clear()
            ctx, self._notify_ctx = self._notify_ctx, None
        if ctx is not None:
            try:
                ctx.term()
            except Exception:
                pass
        for key in list(self._pending_keys):
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
            except Exception:
                pass
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
        self._pending_keys.clear()

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "chunk_notify": self._chunk_notify_enabled,
            "chunk_notify_dropped": self._notify_dropped,
            **self._metrics,
        }
