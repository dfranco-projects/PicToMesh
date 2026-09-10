"""
API tests
=========
Uses fakeredis so no real Redis server is needed.
Tests cover: health check, job creation, job status, mesh serving.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fake_aioredis
import pytest
from arq.connections import ArqRedis
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes.jobs import MAX_FILES, MAX_UPLOAD_BYTES
from backend.models import JobStatus

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_arq():
    arq = MagicMock(spec=ArqRedis)
    arq.enqueue_job = AsyncMock(return_value=None)
    arq.aclose = AsyncMock()
    return arq


@pytest.fixture()
def client(fake_arq, tmp_path):
    """TestClient with fakeredis injected — no real Redis required."""
    fake_redis = fake_aioredis.FakeRedis(decode_responses=True)

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
