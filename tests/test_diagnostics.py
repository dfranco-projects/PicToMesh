"""
Startup failure diagnosis tests
===============================
The model loaders hide why a download failed, so the worker asks the hub
itself. These tests fake the hub's answer; no network is used.
"""

import httpx
import huggingface_hub.constants
import huggingface_hub.utils
import pytest

from backend.worker.diagnostics import describe_startup_failure

ENDPOINT = "https://huggingface.co"


class _FakeSession:
    def __init__(self, outcome: int | Exception):
        self.outcome = outcome

    def head(self, url: str, timeout: float) -> httpx.Response:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return httpx.Response(self.outcome, request=httpx.Request("HEAD", url))


@pytest.fixture
def hub(monkeypatch):
    """Set what a HEAD request to the hub returns (a status code) or raises."""
    monkeypatch.setattr(huggingface_hub.constants, "ENDPOINT", ENDPOINT)

    def answer(outcome: int | Exception) -> None:
        monkeypatch.setattr(huggingface_hub.utils, "get_session", lambda: _FakeSession(outcome))

    return answer


def _failed_download() -> OSError:
    """What transformers raises when a download fails: the network error is not in it."""
    try:
        try:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        except RuntimeError as closed:
            raise OSError(
                "Can't load image processor for 'depth-anything/Depth-Anything-V2-Small-hf'."
            ) from closed
    except OSError as e:
        return e


def test_certificate_failure_is_named(hub):
    hub(httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))

    message, detail = describe_startup_failure(_failed_download())

    assert message == (
        "The AI models couldn't be downloaded: the certificate check for huggingface.co failed."
    )
    assert "Can't load image processor" in detail
    assert "client has been closed" in detail
    assert "CERTIFICATE_VERIFY_FAILED" in detail


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (httpx.ConnectError("[Errno 8] nodename nor servname provided"), "couldn't be reached"),
        (httpx.ConnectTimeout("timed out"), "timed out"),
        (403, "answered HTTP 403"),
    ],
)
def test_hub_problems_are_named(hub, outcome, reason):
    hub(outcome)

    message, _ = describe_startup_failure(_failed_download())

    assert message.startswith("The AI models couldn't be downloaded: ")
    assert reason in message


def test_offline_mode_names_the_missing_cache_instead_of_probing(hub, monkeypatch):
    """HF_HUB_OFFLINE makes huggingface_hub refuse every request, the probe included."""
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    hub(AssertionError("the hub must not be probed"))

    message, detail = describe_startup_failure(_failed_download())

    assert message == (
        "The AI models couldn't be downloaded: "
        "HF_HUB_OFFLINE is set and some are missing from the local cache."
    )
    assert "HF_HUB_OFFLINE=1" in detail


def test_reachable_hub_means_the_models_failed_to_load(hub):
    hub(200)

    message, detail = describe_startup_failure(_failed_download())

    assert message == "The AI models couldn't be loaded."
    assert "Hub check" not in detail


def test_non_io_failure_does_not_probe_the_hub(hub):
    hub(AssertionError("the hub must not be probed"))

    message, detail = describe_startup_failure(RuntimeError("MPS backend out of memory"))

    assert message == "The AI models couldn't be loaded."
    assert detail == "RuntimeError: MPS backend out of memory"
