# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Event-driven chunk notification for the SHM connector (RFC #6870 task B1).

Two layers are covered:

- connector level: the PUSH/PULL notification channel itself (wakeups,
  HWM overflow handling, endpoint naming, opt-in resolution);
- adapter level: the recv loop sleeping on notifications instead of the
  fixed 1 ms backoff, and the fallback poll that recovers a lost hint.

The SHM data plane is deliberately not exercised here (see
``test_shm_connector.py``); these tests only pin the hint semantics that
must never affect correctness.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.request import RequestStatus

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
    _notify_endpoint,
)
from vllm_omni.distributed.omni_connectors.transfer_adapter.base import (
    OmniTransferAdapterBase,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _notify_salt(monkeypatch):
    """Isolate notification endpoints between test cases."""
    monkeypatch.setenv("VLLM_OMNI_CHUNK_NOTIFY_SALT", f"test{time.monotonic_ns()}")


@pytest.fixture()
def sender_connector():
    c = SharedMemoryConnector({"stage_id": 0, "chunk_notify": True})
    yield c
    c.close()


@pytest.fixture()
def receiver_connector():
    c = SharedMemoryConnector({"stage_id": 1, "chunk_notify": True})
    yield c
    c.close()


# ── Connector level: opt-in resolution ──────────────────────────────


class TestNotifyOptIn:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_OMNI_CHUNK_NOTIFY", raising=False)
        connector = SharedMemoryConnector({})

        assert not connector._chunk_notify_enabled
        assert not connector.chunk_notify_enabled()
        connector.close()

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv("VLLM_OMNI_CHUNK_NOTIFY", "1")
        connector = SharedMemoryConnector({})

        assert connector.chunk_notify_enabled()
        connector.close()

    def test_extra_overrides_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_OMNI_CHUNK_NOTIFY", "1")
        connector = SharedMemoryConnector({"chunk_notify": False})

        assert not connector.chunk_notify_enabled()
        connector.close()

    def test_endpoint_includes_edge_and_salt(self, monkeypatch):
        monkeypatch.setenv("VLLM_OMNI_CHUNK_NOTIFY_SALT", "abc")

        assert _notify_endpoint("0", "1").endswith("_0_1_abc")
        assert _notify_endpoint("2", "3") != _notify_endpoint("0", "1")


# ── Connector level: hint semantics ─────────────────────────────────


class TestNotifyChannel:
    def test_put_hint_wakes_wait_immediately(self, sender_connector, receiver_connector):
        # Register the receive edge first (the recv loop's get() does this
        # before its first wait).
        receiver_connector.get("0", "1", "unrelated-key")
        # Let the PULL connect settle; a hint sent before the connect is
        # only queued, not lost, so this is robustness not a requirement.
        time.sleep(0.05)

        sender_connector.notify_chunk_put("0", "1", "req-key")
        start = time.monotonic()

        receiver_connector.wait_for_chunk_notify(timeout_ms=5000)

        elapsed = time.monotonic() - start
        assert elapsed < 2.5, f"wait should be woken by the hint, took {elapsed:.3f}s"

    def test_wait_times_out_without_hint(self, receiver_connector):
        receiver_connector.get("0", "1", "unrelated-key")
        start = time.monotonic()

        receiver_connector.wait_for_chunk_notify(timeout_ms=300)

        elapsed = time.monotonic() - start
        assert elapsed >= 0.25, f"wait should sleep for the timeout, took {elapsed:.3f}s"

    def test_wait_without_receive_edge_sleeps(self, receiver_connector):
        # No get() yet: no PULL exists. Must not degenerate into a busy
        # spin (immediate return).
        start = time.monotonic()

        receiver_connector.wait_for_chunk_notify(timeout_ms=300)

        elapsed = time.monotonic() - start
        assert elapsed >= 0.25, f"wait should sleep, returned after {elapsed:.3f}s"

    def test_hwm_overflow_drops_silently(self, sender_connector):
        # Sender-side HWM is 1024 and no receiver ever connects/reads:
        # every send past the HWM must be dropped, never raise, and never
        # block the put() critical path.
        for i in range(3000):
            sender_connector.notify_chunk_put("0", "1", f"key-{i}")

        assert sender_connector._notify_dropped > 0
        assert sender_connector._metrics["puts"] == 0  # data plane untouched

    def test_hint_after_drain_still_wakes(self, sender_connector, receiver_connector):
        receiver_connector.get("0", "1", "unrelated-key")
        time.sleep(0.05)
        # A first hint is consumed by a wait, a later hint must still wake.
        sender_connector.notify_chunk_put("0", "1", "req-a")
        receiver_connector.wait_for_chunk_notify(timeout_ms=5000)

        sender_connector.notify_chunk_put("0", "1", "req-b")
        start = time.monotonic()
        receiver_connector.wait_for_chunk_notify(timeout_ms=5000)

        assert time.monotonic() - start < 2.5

    def test_health_reports_notify_state(self, sender_connector, receiver_connector):
        health = sender_connector.health()

        assert health["chunk_notify"] is True
        assert health["chunk_notify_dropped"] == 0


# ── Adapter level: recv-loop integration ────────────────────────────


class _NotifyFakeConnector:
    """Connector stand-in exposing only the notification protocol.

    ``wait_for_chunk_notify`` sleeps on an event so the test controls
    wakeups deterministically without ZMQ or SHM.
    """

    supports_chunk_notify = True
    stage_id = 1

    def __init__(self):
        self.wake_event = threading.Event()
        self.wait_calls = 0
        self.pending_payload = None
        self.get_calls = 0

    def chunk_notify_enabled(self):
        return True

    def notify_chunk_put(self, from_stage, to_stage, put_key):
        pass

    def wait_for_chunk_notify(self, timeout_ms, wake_fds=()):
        self.wait_calls += 1
        self.wake_event.wait(timeout_ms / 1000.0)
        self.wake_event.clear()

    def get(self, from_stage, to_stage, get_key, metadata=None):
        self.get_calls += 1
        return self.pending_payload

    def close(self):
        pass


def _req(req_id: str):
    request = Mock(
        client_index=0,
        request_id=req_id,
        external_req_id=req_id,
        status=RequestStatus.WAITING,
        prompt_token_ids=[],
        num_prompt_tokens=0,
        num_computed_tokens=0,
        num_output_placeholders=0,
        prefill_stats=None,
        additional_information=None,
        resumable=False,
    )
    request.is_finished = lambda: RequestStatus.is_finished(request.status)
    return request


def _build_adapter(monkeypatch, connector, **env):
    """A real ``OmniTransferAdapterBase`` driving the fake connector.

    The base recv/save loops run unmodified; ``_poll_single_request`` is
    overridden per-instance with a one-payload pseudo chunk poll.
    """

    def _poll(entry):
        result = connector.get("0", "1", entry.request_id)
        if result is None:
            return False
        payload, _size = result
        if payload.get("meta", {}).get("finished"):
            entry.request.resumable = False
        adapter._finished_load_reqs.add(entry.request_id)
        return True

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    adapter = OmniTransferAdapterBase(config=None)
    adapter.connector = connector
    adapter._poll_single_request = _poll
    return adapter


def _register(adapter, request):
    """Admit a request the way the chunk adapter's load_async does."""
    adapter.request_ids_mapping[request.request_id] = request.external_req_id
    with adapter._recv_cond:
        adapter._pending_load_reqs.append(request)
        adapter._recv_cond.notify()


def _wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


class TestRecvLoopNotify:
    def test_chunk_arrival_wakes_via_notification(self, monkeypatch):
        connector = _NotifyFakeConnector()
        # Fallback far beyond the test budget: success can only come from
        # the notification wakeup, never from the timeout.
        adapter = _build_adapter(
            monkeypatch, connector, VLLM_OMNI_CHUNK_NOTIFY_FALLBACK_MS="10000"
        )
        try:
            _register(adapter, _req("req-1"))
            assert _wait_until(lambda: connector.get_calls >= 1)
            # The chunk lands and the sender-side hint fires.
            connector.pending_payload = ({"meta": {"finished": False}}, 1)
            connector.wake_event.set()

            assert _wait_until(lambda: "req-1" in adapter._finished_load_reqs)
            assert connector.wait_calls >= 1
        finally:
            adapter.shutdown()
            connector.wake_event.set()
            adapter.recv_thread.join(timeout=2.0)

    def test_lost_hint_recovered_by_fallback(self, monkeypatch):
        connector = _NotifyFakeConnector()
        # No hint will ever fire (wake_event stays clear): the fallback
        # timeout is the only wakeup, and it must still deliver the chunk.
        adapter = _build_adapter(
            monkeypatch, connector, VLLM_OMNI_CHUNK_NOTIFY_FALLBACK_MS="50"
        )
        try:
            _register(adapter, _req("req-2"))
            assert _wait_until(lambda: connector.get_calls >= 1)
            connector.pending_payload = ({"meta": {"finished": False}}, 1)

            assert _wait_until(lambda: "req-2" in adapter._finished_load_reqs)
        finally:
            adapter.shutdown()
            connector.wake_event.set()
            adapter.recv_thread.join(timeout=2.0)

    def test_shutdown_exits_within_fallback(self, monkeypatch):
        connector = _NotifyFakeConnector()
        adapter = _build_adapter(
            monkeypatch, connector, VLLM_OMNI_CHUNK_NOTIFY_FALLBACK_MS="200"
        )
        _register(adapter, _req("req-3"))
        time.sleep(0.2)
        assert adapter.recv_thread.is_alive()

        adapter.shutdown()

        adapter.recv_thread.join(timeout=2.0)
        assert not adapter.recv_thread.is_alive()


class TestRecvLoopConfig:
    def test_connectorless_adapter_not_active(self, monkeypatch):
        monkeypatch.delenv("VLLM_OMNI_CHUNK_NOTIFY", raising=False)
        adapter = OmniTransferAdapterBase(config=None)
        try:
            assert not adapter._chunk_notify_active()
        finally:
            adapter.shutdown()

    def test_backoff_env_parsed(self, monkeypatch):
        from vllm_omni.distributed.omni_connectors.transfer_adapter import base as adapter_base

        monkeypatch.setenv("VLLM_OMNI_CHUNK_RECV_BACKOFF_MS", "0.5")
        monkeypatch.delenv("VLLM_OMNI_CHUNK_NOTIFY_FALLBACK_MS", raising=False)

        assert adapter_base._parse_positive_ms(adapter_base._RECV_BACKOFF_ENV, 1.0) == 0.5

    def test_backoff_env_invalid_falls_back(self, monkeypatch):
        from vllm_omni.distributed.omni_connectors.transfer_adapter import base as adapter_base

        monkeypatch.setenv("VLLM_OMNI_CHUNK_RECV_BACKOFF_MS", "not-a-number")

        assert adapter_base._parse_positive_ms(adapter_base._RECV_BACKOFF_ENV, 1.0) == 1.0

    def test_notify_supported_connector_detected(self):
        connector = _NotifyFakeConnector()
        adapter = SimpleNamespace(connector=connector)

        assert OmniTransferAdapterBase._chunk_notify_active(adapter)

    def test_unsupported_connector_not_active(self):
        connector = SimpleNamespace()  # no supports_chunk_notify attribute
        adapter = SimpleNamespace(connector=connector)

        assert not OmniTransferAdapterBase._chunk_notify_active(adapter)

    def test_supported_but_disabled_not_active(self):
        connector = _NotifyFakeConnector()
        connector.chunk_notify_enabled = lambda: False
        adapter = SimpleNamespace(connector=connector)

        assert not OmniTransferAdapterBase._chunk_notify_active(adapter)
