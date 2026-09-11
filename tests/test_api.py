"""
API tests
=========
Uses fakeredis so no real Redis server is needed.
Tests cover: health check, job creation, job status, mesh serving.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import fakeredis.aioredis as fake_aioredis
import pytest
from arq.connections import ArqRedis
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes.jobs import MAX_FILES, MAX_UPLOAD_BYTES
from backend.config import WORKER_ERROR_KEY, WORKER_HEARTBEAT_KEY
from backend.models import JobStatus, WorkerError, WorkerState, WorkerStatus

CERT_FAILURE = (
    "The AI models couldn't be downloaded: the certificate check for huggingface.co failed."
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_arq():
    arq = MagicMock(spec=ArqRedis)
    arq.enqueue_job = AsyncMock(return_value=None)
    arq.aclose = AsyncMock()
    return arq


@pytest.fixture()
def redis_server():
    return fakeredis.FakeServer()


@pytest.fixture()
def worker_keys(redis_server):
    """Sync handle on the app's Redis for setting worker state; the worker starts ready."""
    keys = fakeredis.FakeRedis(server=redis_server, decode_responses=True)
    keys.set(WORKER_HEARTBEAT_KEY, WorkerStatus(state=WorkerState.ready).model_dump_json())
    return keys


def set_worker(keys, heartbeat: WorkerState | None, error: str | None = None) -> None:
    """Put the worker's Redis keys into a given state; None removes a key."""
    if heartbeat is None:
        keys.delete(WORKER_HEARTBEAT_KEY)
    else:
        keys.set(WORKER_HEARTBEAT_KEY, WorkerStatus(state=heartbeat).model_dump_json())
    if error is None:
        keys.delete(WORKER_ERROR_KEY)
    else:
        at = datetime(2026, 9, 11, 21, 1, tzinfo=timezone.utc)
        failure = WorkerError(message=error, detail="OSError: Can't load image processor", at=at)
        keys.set(WORKER_ERROR_KEY, failure.model_dump_json())


@pytest.fixture()
def client(fake_arq, redis_server, worker_keys, tmp_path):
    """TestClient with fakeredis injected — no real Redis required."""
    fake_redis = fake_aioredis.FakeRedis(server=redis_server, decode_responses=True)

    @asynccontextmanager
    async def _fake_lifespan(a):
        a.state.redis = fake_redis
        a.state.arq = fake_arq
        yield
        await fake_redis.aclose()

    with patch.object(app.router, "lifespan_context", _fake_lifespan):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── health ────────────────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("heartbeat", "error", "state", "has_error"),
    [
        (WorkerState.ready, None, "ready", False),
        (WorkerState.ready, CERT_FAILURE, "ready", False),  # error about to be cleared
        (WorkerState.loading, None, "loading", False),
        (WorkerState.loading, CERT_FAILURE, "loading", True),  # retrying after a failure
        (None, CERT_FAILURE, "failed", True),
        (None, None, "offline", False),
    ],
)
def test_worker_health_reports_state(client, worker_keys, heartbeat, error, state, has_error):
    set_worker(worker_keys, heartbeat, error)

    r = client.get("/health/worker")

    assert r.status_code == 200
    body = r.json()
    assert body["state"] == state
    assert (body["error"] is not None) == has_error
    if has_error:
        assert body["error"]["message"] == CERT_FAILURE


# ── POST /jobs ────────────────────────────────────────────────────────────────


def test_create_job_returns_202(client, fake_arq):
    img_bytes = b"\xff\xd8\xff" + b"\x00" * 100
    r = client.post(
        "/jobs",
        files=[("files", ("photo.jpg", io.BytesIO(img_bytes), "image/jpeg"))],
    )
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body
    assert body["status"] == JobStatus.queued


def test_create_job_enqueues_task(client, fake_arq):
    img_bytes = b"\xff\xd8\xff" + b"\x00" * 100
    client.post(
        "/jobs",
        files=[("files", ("photo.jpg", io.BytesIO(img_bytes), "image/jpeg"))],
    )
    fake_arq.enqueue_job.assert_awaited_once()
    call_args = fake_arq.enqueue_job.call_args
    assert call_args.args[0] == "process_images"


def test_create_job_no_files_returns_422(client):
    r = client.post("/jobs", files=[])
    assert r.status_code == 422


def test_upload_filename_cannot_escape_the_upload_dir(client, fake_arq, tmp_path, monkeypatch):
    """A traversing multipart filename must not write outside media_dir."""
    from backend import config

    monkeypatch.setattr(config.settings, "media_dir", tmp_path)
    escape_target = tmp_path.parent / "escaped.jpg"

    r = client.post(
        "/jobs",
        files=[
            (
                "files",
                ("../../escaped.jpg", io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 32), "image/jpeg"),
            )
        ],
    )

    assert r.status_code == 202
    assert not escape_target.exists()
    saved = [Path(p) for p in fake_arq.enqueue_job.call_args.args[2]]
    for path in saved:
        assert tmp_path in path.resolve().parents


def test_upload_keeps_only_allowed_extensions(client, fake_arq, tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config.settings, "media_dir", tmp_path)
    client.post(
        "/jobs",
        files=[("files", ("payload.py", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg"))],
    )
    saved = Path(fake_arq.enqueue_job.call_args.args[2][0])
    assert saved.suffix == ".jpg"


def test_create_job_rejects_too_many_files(client):
    files = [
        ("files", (f"{i}.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg"))
        for i in range(MAX_FILES + 1)
    ]
    r = client.post("/jobs", files=files)
    assert r.status_code == 413


def test_create_job_rejects_oversized_file(client, tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config.settings, "media_dir", tmp_path)
    oversized = b"\xff\xd8\xff" + b"\x00" * (MAX_UPLOAD_BYTES + 1)
    r = client.post(
        "/jobs",
        files=[("files", ("big.jpg", io.BytesIO(oversized), "image/jpeg"))],
    )
    assert r.status_code == 413
    assert not any(tmp_path.iterdir())


def test_create_job_without_worker_returns_503(
    client, fake_arq, worker_keys, tmp_path, monkeypatch
):
    """With no worker the job would sit queued forever, so refuse it up front."""
    from backend import config

    monkeypatch.setattr(config.settings, "media_dir", tmp_path)
    set_worker(worker_keys, None)

    r = client.post(
        "/jobs",
        files=[("files", ("a.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg"))],
    )

    assert r.status_code == 503
    assert r.json()["detail"] == "The processing worker isn't running. Restart PicToMesh."
    fake_arq.enqueue_job.assert_not_awaited()
    assert not any(tmp_path.iterdir())


def test_create_job_after_failed_startup_explains_why(client, fake_arq, worker_keys):
    set_worker(worker_keys, None, CERT_FAILURE)

    r = client.post(
        "/jobs",
        files=[("files", ("a.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg"))],
    )

    assert r.status_code == 503
    assert r.json()["detail"].startswith(CERT_FAILURE)
    fake_arq.enqueue_job.assert_not_awaited()


def test_create_job_while_models_load_is_accepted(client, fake_arq, worker_keys):
    """A loading worker picks the job up once its models are ready."""
    set_worker(worker_keys, WorkerState.loading)

    r = client.post(
        "/jobs",
        files=[("files", ("a.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg"))],
    )

    assert r.status_code == 202
    fake_arq.enqueue_job.assert_awaited_once()


def test_create_job_rejects_unknown_format(client):
    r = client.post(
        "/jobs?fmt=exe",
        files=[("files", ("a.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg"))],
    )
    assert r.status_code == 422


# ── GET /jobs/{id} ────────────────────────────────────────────────────────────


def test_get_job_not_found(client):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.not_found)):
        r = client.get("/jobs/nonexistent-id")
    assert r.status_code == 404


def test_get_job_queued(client):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.queued)):
        r = client.get("/jobs/some-job-id")
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.queued


@pytest.mark.parametrize(
    ("heartbeat", "error", "expected_error"),
    [
        (None, None, "The processing worker isn't running. Restart PicToMesh."),
        (None, CERT_FAILURE, CERT_FAILURE),
    ],
)
def test_get_job_queued_without_worker_fails(client, worker_keys, heartbeat, error, expected_error):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    set_worker(worker_keys, heartbeat, error)
    with patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.queued)):
        r = client.get("/jobs/some-job-id")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == JobStatus.failed
    assert body["error"].startswith(expected_error)


def test_get_job_queued_while_models_load_stays_queued(client, worker_keys):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    set_worker(worker_keys, WorkerState.loading, CERT_FAILURE)
    with patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.queued)):
        r = client.get("/jobs/some-job-id")

    assert r.json()["status"] == JobStatus.queued


def test_get_job_in_progress_without_heartbeat_stays_in_progress(client, worker_keys):
    """A busy worker can miss heartbeats; the job it holds must not be reported failed."""
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    set_worker(worker_keys, None)
    with patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.in_progress)):
        r = client.get("/jobs/some-job-id")

    assert r.json()["status"] == JobStatus.in_progress


def test_get_job_complete_has_mesh_url(client):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with (
        patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.complete)),
        patch.object(
            Job,
            "result",
            new=AsyncMock(return_value={"status": "complete", "mesh_url": "/meshes/abc/mesh.glb"}),
        ),
    ):
        r = client.get("/jobs/abc")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == JobStatus.complete
    assert body["mesh_url"] == "/meshes/abc/mesh.glb"


def test_get_job_error_result_maps_to_failed(client):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with (
        patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.complete)),
        patch.object(
            Job,
            "result",
            new=AsyncMock(return_value={"status": "failed", "error": "boom"}),
        ),
    ):
        r = client.get("/jobs/abc")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == JobStatus.failed
    assert body["error"] == "boom"


def test_get_job_raising_result_returns_failed(client):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with (
        patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.complete)),
        patch.object(Job, "result", new=AsyncMock(side_effect=RuntimeError("task exploded"))),
    ):
        r = client.get("/jobs/abc")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == JobStatus.failed
    assert "task exploded" in body["error"]


# ── GET /jobs/{id}/stream ─────────────────────────────────────────────────────


def test_stream_unknown_job_returns_404(client):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.not_found)):
        r = client.get("/jobs/nope/stream")
    assert r.status_code == 404


def test_stream_closes_immediately_when_job_already_finished(client):
    """Nothing more gets published for a finished job, so the stream must not wait."""
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with (
        patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.complete)),
        patch.object(
            Job,
            "result",
            new=AsyncMock(return_value={"status": "complete", "mesh_url": "/meshes/abc/mesh.glb"}),
        ),
    ):
        r = client.get("/jobs/abc/stream")

    assert r.status_code == 200
    assert "complete" in r.text
    assert r.text.startswith("data: ")


def test_stream_reports_failure_reason_for_finished_job(client):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    with (
        patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.complete)),
        patch.object(
            Job,
            "result",
            new=AsyncMock(return_value={"status": "failed", "error": "no readable images"}),
        ),
    ):
        r = client.get("/jobs/abc/stream")

    assert r.status_code == 200
    assert "no readable images" in r.text
    assert "failed" in r.text


def test_stream_closes_with_failure_when_queued_job_has_no_worker(client, worker_keys):
    from arq.jobs import Job
    from arq.jobs import JobStatus as ArqJobStatus

    set_worker(worker_keys, None)
    with patch.object(Job, "status", new=AsyncMock(return_value=ArqJobStatus.queued)):
        r = client.get("/jobs/abc/stream")

    assert r.status_code == 200
    assert r.text.startswith("data: ")
    assert "failed" in r.text
    assert "worker isn't running" in r.text


# ── GET /meshes/{job_id}/{filename} ───────────────────────────────────────────


def test_get_mesh_not_found(client):
    r = client.get("/meshes/no-such-job/mesh.glb")
    assert r.status_code == 404


def test_get_mesh_serves_file(client, tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config.settings, "media_dir", tmp_path)

    mesh_dir = tmp_path / "job123"
    mesh_dir.mkdir()
    mesh_file = mesh_dir / "mesh.glb"
    mesh_file.write_bytes(b"GLB")

    r = client.get("/meshes/job123/mesh.glb")
    assert r.status_code == 200
    assert r.content == b"GLB"
