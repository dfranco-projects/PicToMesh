"""Explain a failed worker startup in words a user can act on.

Model loaders hide why a download failed: transformers reports "Can't load
image processor" whether the hub is unreachable, refuses the request or fails
the certificate check. So after a failure, ask the hub directly, through
huggingface_hub's own HTTP session and therefore the same certificate settings.
"""

from __future__ import annotations

import httpx

_DOWNLOAD_FAILED = "The AI models couldn't be downloaded"
_LOAD_FAILED = "The AI models couldn't be loaded."


def describe_startup_failure(exc: BaseException) -> tuple[str, str]:
    """A one-sentence message for the UI and technical detail for bug reports."""
    chain = _chain(exc)
    detail = "\n".join(_describe(e) for e in chain)
    if not any(isinstance(e, OSError | httpx.HTTPError) for e in chain):
        return _LOAD_FAILED, detail
    problem = _probe_hub()
    if problem is None:
        return _LOAD_FAILED, detail
    reason, probe_detail = problem
    return f"{_DOWNLOAD_FAILED}: {reason}.", f"{detail}\nHub check: {probe_detail}"


def _probe_hub() -> tuple[str, str] | None:
    """Why the model hub can't be used right now, or None if it answers."""
    from huggingface_hub import constants
    from huggingface_hub.utils import get_session

    # Offline mode blocks every request, the probe included.
    if constants.HF_HUB_OFFLINE:
        return "HF_HUB_OFFLINE is set and some are missing from the local cache", "HF_HUB_OFFLINE=1"
    host = httpx.URL(constants.ENDPOINT).host
    try:
        response = get_session().head(constants.ENDPOINT, timeout=10)
    except httpx.TimeoutException as e:
        return f"the connection to {host} timed out", _describe(e)
    except httpx.HTTPError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            return f"the certificate check for {host} failed", _describe(e)
        return f"{host} couldn't be reached", _describe(e)
    if response.status_code >= 400:
        return (
            f"{host} answered HTTP {response.status_code}",
            f"HEAD {constants.ENDPOINT} returned {response.status_code}",
        )
    return None


def _chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
