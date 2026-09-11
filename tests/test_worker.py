"""
Worker task tests
=================
Covers image loading with the resolution cap, the status the worker reports
while it loads (heartbeat, readiness, startup failure), and that jobs run off
the event loop. Pipeline execution itself is exercised by the pipeline unit
and E2E tests.
"""

import asyncio
import threading
from pathlib import Path

import cv2
import fakeredis.aioredis as fake_aioredis
import numpy as np
import pytest

from backend.config import WORKER_ERROR_KEY, WORKER_HEARTBEAT_KEY
from backend.models import JobStatus, WorkerError, WorkerState, WorkerStatus
from backend.worker import tasks
from backend.worker.tasks import (
    _MAX_SIDE,
    _heartbeat,
    _load_image,
    process_images,
    shutdown,
    startup,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"  # arq runs on asyncio


def _write_image(path: Path, w: int, h: int) -> str:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_load_image_downscales_to_max_side(tmp_path):
    p = _write_image(tmp_path / "big.png", 2048, 1024)
    img = _load_image(p)
    assert img is not None
    assert img.shape[1] == _MAX_SIDE
    assert img.shape[0] == _MAX_SIDE // 2  # aspect ratio preserved


def test_load_image_keeps_small_images(tmp_path):
    p = _write_image(tmp_path / "small.png", 100, 80)
    img = _load_image(p)
    assert img is not None
    assert img.shape[:2] == (80, 100)


def test_load_image_unreadable_returns_none(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"nope")
    assert _load_image(str(bad)) is None


# ── heartbeat ─────────────────────────────────────────────────────────────────


async def _heartbeat_state(redis) -> WorkerState:
    return WorkerStatus.model_validate_json(await redis.get(WORKER_HEARTBEAT_KEY)).state


@pytest.mark.anyio
async def test_heartbeat_publishes_state_in_an_expiring_key():
    """The key must expire on its own, or a crashed worker would look alive forever."""
    redis = fake_aioredis.FakeRedis()
    ctx = {"redis": redis, "status": WorkerStatus(state=WorkerState.loading)}
    task = asyncio.create_task(_heartbeat(ctx))
    await asyncio.sleep(0.05)

    assert await _heartbeat_state(redis) == WorkerState.loading
    assert 0 < await redis.ttl(WORKER_HEARTBEAT_KEY) <= 60

    task.cancel()
    await redis.aclose()


@pytest.mark.anyio
async def test_shutdown_stops_heartbeat_and_clears_key():
    redis = fake_aioredis.FakeRedis()
    ctx = {"redis": redis, "status": WorkerStatus(state=WorkerState.ready)}
    task = asyncio.create_task(_heartbeat(ctx))
    await asyncio.sleep(0.05)

    await shutdown({"redis": redis, "heartbeat": task})
    await asyncio.sleep(0)

    assert task.cancelled()
    assert not await redis.exists(WORKER_HEARTBEAT_KEY)
    await redis.aclose()


@pytest.mark.anyio
async def test_shutdown_after_failed_startup():
    """arq calls on_shutdown even when on_startup raised, so no heartbeat exists yet."""
    redis = fake_aioredis.FakeRedis()
    await shutdown({"redis": redis})
    assert not await redis.exists(WORKER_HEARTBEAT_KEY)
    await redis.aclose()


# ── startup ───────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_startup_reports_ready_and_clears_the_last_failure(monkeypatch):
    redis = fake_aioredis.FakeRedis()
    await redis.set(WORKER_ERROR_KEY, "a failure from an earlier attempt")
    pipeline = object()
    monkeypatch.setattr(tasks, "_build_pipeline", lambda: pipeline)
    ctx = {"redis": redis}

    await startup(ctx)

    assert ctx["pipeline"] is pipeline
    assert await _heartbeat_state(redis) == WorkerState.ready
    assert not await redis.exists(WORKER_ERROR_KEY)
    await shutdown(ctx)
    await redis.aclose()


@pytest.mark.anyio
async def test_startup_failure_records_the_reason_and_exits(monkeypatch):
    """The UI reads the recorded reason after the worker process is gone."""

    def fail():
        raise OSError("Can't load image processor for 'depth-anything/Depth-Anything-V2-Small-hf'")

    redis = fake_aioredis.FakeRedis()
    monkeypatch.setattr(tasks, "_build_pipeline", fail)
    monkeypatch.setattr(
        tasks,
        "describe_startup_failure",
        lambda exc: ("The AI models couldn't be loaded.", f"OSError: {exc}"),
    )
    ctx = {"redis": redis}

    with pytest.raises(SystemExit) as exit_info:
        await startup(ctx)
    await shutdown(ctx)  # arq runs this after a failed startup
    await asyncio.sleep(0)

    assert exit_info.value.code == 1
    error = WorkerError.model_validate_json(await redis.get(WORKER_ERROR_KEY))
    assert error.message == "The AI models couldn't be loaded."
    assert "depth-anything" in error.detail
    assert not await redis.exists(WORKER_HEARTBEAT_KEY)
    assert ctx["heartbeat"].cancelled()
    await redis.aclose()


# ── process_images ────────────────────────────────────────────────────────────


class _ThreadRecordingPipeline:
    def __init__(self):
        self.thread_id: int | None = None

    def run(self, images, output_path, fmt="glb"):
        self.thread_id = threading.get_ident()
        return output_path.with_suffix(f".{fmt}")


@pytest.mark.anyio
async def test_process_images_runs_pipeline_off_the_event_loop(tmp_path, monkeypatch):
    """A blocked event loop stops the heartbeat, making a busy worker look dead."""
    from backend import config

    monkeypatch.setattr(config.settings, "media_dir", tmp_path)
    redis = fake_aioredis.FakeRedis()
    pipeline = _ThreadRecordingPipeline()
    image = _write_image(tmp_path / "a.png", 64, 64)

    result = await process_images({"redis": redis, "pipeline": pipeline}, "job1", [image])

    assert result["status"] == JobStatus.complete
    assert pipeline.thread_id is not None
    assert pipeline.thread_id != threading.get_ident()
    await redis.aclose()
