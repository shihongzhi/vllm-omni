# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64

import numpy as np
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input import MiniCPMO45PcmAppendBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def pcm_payload(samples: int, *, speech: bool = True) -> dict[str, object]:
    audio = np.ones(samples, dtype=np.float32).tobytes()
    return {
        "type": "audio",
        "audio": base64.b64encode(audio).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
        "is_speech": speech,
    }


def video_payload(samples: int, frames: list[str], *, speech: bool = True) -> dict[str, object]:
    payload = pcm_payload(samples, speech=speech)
    payload["video_frames"] = frames
    return payload


def test_commit_does_not_add_silence_after_incremental_audio_was_drained():
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted = buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    committed = buffer.commit(chunk_period_ms=1_000)

    assert emitted is not None
    assert not buffer.has_pending()
    assert committed is None


def test_append_emits_one_model_unit_when_multiple_units_are_buffered():
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted = buffer.append(pcm_payload(32_000), chunk_period_ms=1_000)

    assert emitted is not None
    assert len(base64.b64decode(emitted["audio"])) == 16_000 * 4
    assert buffer.pending_byte_count == 16_000 * 4


def test_speech_marker_does_not_leak_across_irregular_chunk_boundaries():
    buffer = MiniCPMO45PcmAppendBuffer()

    assert buffer.append(pcm_payload(15_000), chunk_period_ms=1_000) is None
    speech_unit = buffer.append(pcm_payload(2_000, speech=False), chunk_period_ms=1_000)
    silence_unit = buffer.append(pcm_payload(15_000, speech=False), chunk_period_ms=1_000)

    assert speech_unit is not None
    assert speech_unit["is_speech"] is True
    assert silence_unit is not None
    assert silence_unit["is_speech"] is False


def test_force_listen_marker_does_not_leak_across_irregular_chunk_boundaries():
    buffer = MiniCPMO45PcmAppendBuffer()
    forced = pcm_payload(15_000, speech=False)
    forced["force_listen"] = True

    assert buffer.append(forced, chunk_period_ms=1_000) is None
    forced_unit = buffer.append(pcm_payload(2_000, speech=False), chunk_period_ms=1_000)
    unforced_unit = buffer.append(pcm_payload(15_000, speech=False), chunk_period_ms=1_000)

    assert forced_unit is not None
    assert forced_unit["force_listen"] is True
    assert unforced_unit is not None
    assert unforced_unit["force_listen"] is False


def test_pcm_append_rollback_restores_per_span_speech_markers():
    buffer = MiniCPMO45PcmAppendBuffer()
    assert buffer.append(pcm_payload(15_000), chunk_period_ms=1_000) is None
    reservation = buffer.prepare_append(
        pcm_payload(2_000, speech=False),
        operation_id="mixed-unit",
        chunk_period_ms=1_000,
    )

    assert reservation is not None
    reservation.rollback()
    first = buffer.append(pcm_payload(15_000, speech=False), chunk_period_ms=1_000)
    second = buffer.flush(chunk_period_ms=1_000)

    assert first is not None
    assert first["is_speech"] is True
    assert second is not None
    assert second["is_speech"] is False


def test_commit_without_speech_does_not_synthesize_terminal_audio():
    buffer = MiniCPMO45PcmAppendBuffer()
    buffer.append(pcm_payload(8_000, speech=False), chunk_period_ms=1_000)

    committed = buffer.commit(chunk_period_ms=1_000)

    assert committed is None


def test_commit_resets_cumulative_turn_accounting():
    buffer = MiniCPMO45PcmAppendBuffer()
    buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    buffer.commit(chunk_period_ms=1_000)

    empty = buffer.commit(chunk_period_ms=1_000)

    assert empty is None


def test_pcm_append_reservation_rollback_restores_emitted_audio():
    buffer = MiniCPMO45PcmAppendBuffer()
    original = pcm_payload(16_000)

    reservation = buffer.prepare_append(
        original,
        operation_id="append-1",
        chunk_period_ms=1_000,
    )

    assert reservation is not None
    assert not buffer.has_pending()
    reservation.rollback()
    assert buffer.has_pending()

    retried = buffer.flush(chunk_period_ms=1_000)
    assert retried is not None
    assert base64.b64decode(retried["audio"]) == base64.b64decode(original["audio"])


def test_pcm_commit_keeps_prior_append_reservation_active():
    buffer = MiniCPMO45PcmAppendBuffer()
    append_reservation = buffer.prepare_append(
        pcm_payload(16_000),
        operation_id="append-before-commit",
        chunk_period_ms=1_000,
    )

    commit_reservation = buffer.prepare_commit(
        operation_id="commit-after-append",
        chunk_period_ms=1_000,
    )

    assert append_reservation is not None
    assert append_reservation.active
    assert commit_reservation.payload is None
    append_reservation.commit()
    commit_reservation.commit()


def test_pcm_commit_reservation_rollback_restores_residual_audio():
    buffer = MiniCPMO45PcmAppendBuffer()
    original = pcm_payload(8_000)
    assert (
        buffer.prepare_append(
            original,
            operation_id="buffer-half-chunk",
            chunk_period_ms=1_000,
        )
        is None
    )

    reservation = buffer.prepare_commit(
        operation_id="final-half-chunk",
        chunk_period_ms=1_000,
    )

    assert reservation.payload is not None
    assert reservation.payload["final"] is True
    reservation.rollback()

    retried = buffer.prepare_commit(
        operation_id="retry-final-half-chunk",
        chunk_period_ms=1_000,
    )
    assert retried.payload is not None
    assert base64.b64decode(retried.payload["audio"]) == (base64.b64decode(original["audio"]) + b"\x00" * (8_000 * 4))


@pytest.mark.parametrize("units", [8, 64])
def test_stacked_pair_drains_fully_each_unit(units):
    """Frames ride the unit-closing append; both must attach to that unit.

    Attaching only one frame per emit left the composite queued forever
    (one leftover frame per second of video) and shifted later units onto
    stale frames.
    """
    buffer = MiniCPMO45PcmAppendBuffer()

    for _ in range(units):
        emitted = buffer.append(video_payload(16_000, ["base", "composite"]), chunk_period_ms=1_000)
        assert emitted is not None
        assert emitted["video_frames"] == ["base", "composite"]

    # A frameless follow-up unit must stay frameless: any leftover frame
    # from the drained units would surface here (FIFO queue).
    followup = buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    assert followup is not None
    assert "video_frames" not in followup


def test_single_frame_per_unit_attaches_to_its_unit():
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted = buffer.append(video_payload(16_000, ["base"]), chunk_period_ms=1_000)
    followup = buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)

    assert emitted is not None
    assert emitted["video_frames"] == ["base"]
    assert followup is not None
    assert "video_frames" not in followup


def test_frames_wait_for_the_unit_they_close():
    """Client cadence: frames ride the append that completes the 1 s unit."""
    buffer = MiniCPMO45PcmAppendBuffer()

    for _ in range(4):
        assert buffer.append(video_payload(3_200, []), chunk_period_ms=1_000) is None

    emitted = buffer.append(video_payload(3_200, ["base", "composite"]), chunk_period_ms=1_000)
    assert emitted is not None
    assert emitted["video_frames"] == ["base", "composite"]
    assert not buffer.has_pending()


def test_commit_drops_frames_of_units_that_never_closed():
    """Unattached frames belong to the committed generation, not the next."""
    buffer = MiniCPMO45PcmAppendBuffer()

    assert buffer.append(video_payload(8_000, ["orphan"], speech=False), chunk_period_ms=1_000) is None
    committed = buffer.commit(chunk_period_ms=1_000)
    assert committed is None  # no speech in the generation -> no terminal flush

    emitted = buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    assert emitted is not None
    assert "video_frames" not in emitted


def test_append_rollback_restores_queued_frames():
    buffer = MiniCPMO45PcmAppendBuffer()

    reservation = buffer.prepare_append(
        video_payload(16_000, ["base", "composite"]),
        operation_id="append-frames",
        chunk_period_ms=1_000,
    )
    assert reservation is not None
    assert reservation.payload is not None
    assert reservation.payload["video_frames"] == ["base", "composite"]
    reservation.rollback()

    retried = buffer.flush(chunk_period_ms=1_000)
    assert retried is not None
    assert retried["video_frames"] == ["base", "composite"]
