"""AnyRouter provider profile for Hermes Agent.

AnyRouter (https://anyrouter.dev) is a unified model gateway with an
OpenAI-style ``/v1`` surface and ``vendor/model`` routing slugs, including
first-party meta-models (``anyrouter/hermes``, ``anyrouter/auto``) that
auto-route across upstreams. This profile wires the request-shaping that
AnyRouter's API documents:

  - ``extra_body.session_id`` — sticky-session id grouping related requests in
    the Request Logs dashboard (body wins over the ``x-session-id`` header).
  - ``extra_body.provider`` — request-level routing preferences (``only`` /
    ``ignore`` / ``order`` / ``sort`` / ``allow_fallbacks`` / ``max_price``).
  - ``extra_body.reasoning`` — reasoning effort passthrough; AnyRouter accepts
    the OpenAI-style ``reasoning`` object natively and translates it to each
    upstream's thinking control, so the profile forwards the caller's intent
    rather than reimplementing per-upstream workarounds.

App-attribution headers (``HTTP-Referer`` / ``X-AnyRouter-Title`` /
``X-AnyRouter-Source`` / ``X-AnyRouter-Categories``) are OFF by default and
are only attached when the user opts in with
``ANYROUTER_APP_ATTRIBUTION=1``. They credit Hermes traffic in AnyRouter's
public app rankings; nothing is tagged unless asked for.

See: https://docs.anyrouter.dev/api-reference/chat-completions and
https://docs.anyrouter.dev/features/app-attribution

Hermes core moves fast and a plugin has no say in which build a user runs, so
the points of contact with ``ProviderProfile`` — the fields we set and the
inherited ``fetch_models`` we delegate to — are resolved by introspection
rather than assumed. See ``_supported_kwargs`` and ``_inherited_fetch``.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

PROVIDER_ID = "anyrouter"
API_KEY_ENV = "ANYROUTER_API_KEY"
BASE_URL_ENV = "ANYROUTER_BASE_URL"
ATTRIBUTION_ENV = "ANYROUTER_APP_ATTRIBUTION"

OPENAI_BASE_URL = "https://anyrouter.dev/api/v1"
MODELS_URL = "https://anyrouter.dev/api/v1/models"
SIGNUP_URL = "https://anyrouter.dev"

# Capability markers in the public catalog that mean "this route accepts tool
# calls". AnyRouter fronts many upstreams; entries advertise
# ``function-calling`` (and sometimes ``tools``). Either qualifies — Hermes is
# an agent, and a model that cannot call tools is dead weight in the picker.
_AGENTIC_CAPABILITIES = frozenset({"function-calling", "tools"})

_CATALOG_CACHE: list[str] | None = None


def _agentic(entry: dict[str, Any]) -> bool:
    """True when a catalog entry advertises tool-calling support."""
    caps = entry.get("capabilities")
    if not isinstance(caps, list):
        return False
    return bool(_AGENTIC_CAPABILITIES & set(caps))


def _attribution_enabled() -> bool:
    """App-attribution headers are opt-in: ``ANYROUTER_APP_ATTRIBUTION=1``."""
    return os.environ.get(ATTRIBUTION_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_headers() -> dict[str, str]:
    """Headers attached to every request. Attribution only when opted in."""
    if not _attribution_enabled():
        return {}
    return {
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-AnyRouter-Title": "Hermes Agent",
        "X-AnyRouter-Source": "cli-agent",
        "X-AnyRouter-Categories": "cli-agent",
    }


def _url_opener():
    """Return Hermes' credential-safe URL opener, or urllib's as a fallback.

    ``hermes_cli.urllib_security.open_credentialed_url`` guards a request that
    carries an Authorization header; builds predating that module fetch their
    own catalogs with a bare ``urllib.request.urlopen``. Falling back to the
    same call keeps the plugin working on those builds without ever being
    laxer than the host build is with its own credentialed catalog fetches.
    """
    try:
        from hermes_cli.urllib_security import open_credentialed_url

        return open_credentialed_url
    except ImportError:
        logger.debug(
            "anyrouter: this Hermes build has no urllib_security helper — "
            "using urllib.request.urlopen, as the build's own catalog fetch does"
        )
        return urllib.request.urlopen


def _supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop profile fields the installed Hermes build does not define.

    ``ProviderProfile`` gains fields over time (``supports_vision`` and
    ``hostname`` are recent). Passing one to an older build raises TypeError
    at import, which the plugin loader swallows — the provider then silently
    fails to register and the user sees "unknown provider" with no cause.
    Declaring the full modern field set and filtering it to what this build
    accepts degrades to a slightly less capable profile instead.
    """
    try:
        known = {f.name for f in dataclasses.fields(cls)}
    except TypeError:  # pragma: no cover — non-dataclass profile base
        return dict(kwargs)
    dropped = sorted(set(kwargs) - known)
    if dropped:
        logger.debug(
            "anyrouter: this Hermes build has no profile field(s) %s — skipping",
            ", ".join(dropped),
        )
    return {k: v for k, v in kwargs.items() if k in known}


class AnyRouterProfile(ProviderProfile):
    """AnyRouter gateway — session, routing-preference, reasoning passthrough."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return tool-capable model ids from the AnyRouter catalog.

        The plain ``/v1/models`` response already carries a ``capabilities``
        list per model, so the picker can be filtered to entries that accept
        tool calls — the only models an agent can use.

        Falls back to the inherited ``/v1/models`` handling whenever the
        catalog is unreachable, malformed, or filters down to nothing — a
        degraded picker beats an empty one.

        A caller-supplied ``base_url`` that differs from the default means the
        user pointed Hermes at a proxy or a self-hosted relay. Its catalog is
        not ours to interpret, so defer to the generic path in that case.
        """
        global _CATALOG_CACHE  # noqa: PLW0603

        caller_base = (base_url or "").strip()
        custom_base = bool(caller_base) and (
            caller_base.rstrip("/") != self.base_url.rstrip("/")
        )
        if custom_base:
            return self._inherited_fetch(
                api_key=api_key, base_url=base_url, timeout=timeout
            )

        if _CATALOG_CACHE is not None:
            return _CATALOG_CACHE

        models = self._fetch_agentic_catalog(api_key=api_key, timeout=timeout)
        if models:
            _CATALOG_CACHE = models
            return models

        return self._inherited_fetch(
            api_key=api_key, base_url=base_url, timeout=timeout
        )

    def _inherited_fetch(
        self, *, api_key: str | None, base_url: str | None, timeout: float
    ) -> list[str] | None:
        """Call the base-class fetch, passing only arguments it accepts.

        ``base_url`` was added to the base signature after this plugin's
        contract was written. Passing it to an older build is a TypeError, and
        the fallback path is exactly where a crash is least affordable — we
        are already here because the primary catalog failed.
        """
        parent = super()
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        try:
            accepted = inspect.signature(parent.fetch_models).parameters
        except (TypeError, ValueError):  # pragma: no cover — C-level callable
            accepted = {}
        if "base_url" in accepted:
            kwargs["base_url"] = base_url
        return parent.fetch_models(**kwargs)

    def _fetch_agentic_catalog(
        self, *, api_key: str | None, timeout: float
    ) -> list[str] | None:
        """Fetch and filter the capability-annotated catalog, or None."""
        open_url = _url_opener()

        req = urllib.request.Request(MODELS_URL)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        for key, value in self.default_headers.items():
            req.add_header(key, value)

        try:
            with open_url(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(anyrouter): catalog API failed: %s", exc)
            return None

        entries = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            logger.debug("fetch_models(anyrouter): unexpected catalog shape")
            return None

        seen: set[str] = set()
        models: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or not _agentic(entry):
                continue
            model_id = entry.get("model_id") or entry.get("id")
            if isinstance(model_id, str) and model_id and model_id not in seen:
                seen.add(model_id)
                models.append(model_id)

        if not models:
            logger.debug("fetch_models(anyrouter): catalog filtered to zero models")
            return None
        return models

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        prefs = context.get("provider_preferences")
        if prefs:
            body["provider"] = prefs
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward the reasoning config as ``extra_body.reasoning``.

        AnyRouter accepts the OpenAI-style ``reasoning`` object natively
        (``effort`` / ``enabled`` / ``exclude`` / ``max_tokens``) and owns the
        translation to each upstream's thinking control, so the profile simply
        passes the caller's intent through.
        """
        extra_body: dict[str, Any] = {}
        if supports_reasoning:
            if reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


PROFILE_FIELDS: dict[str, Any] = {
    "name": PROVIDER_ID,
    "display_name": "AnyRouter",
    "description": "AnyRouter — unified model gateway (OpenAI-compatible)",
    "signup_url": SIGNUP_URL,
    "env_vars": (API_KEY_ENV, BASE_URL_ENV),
    "base_url": OPENAI_BASE_URL,
    "models_url": MODELS_URL,
    "hostname": "anyrouter.dev",
    "auth_type": "api_key",
    "api_mode": "chat_completions",
    # AnyRouter relays OpenAI-compatible multimodal content to whichever
    # upstream serves the model; the catalog marks vision-capable routes.
    "supports_vision": True,
    # First-party meta-model ranked for agentic work, 1M context — suits
    # compression, title generation, and vision auxiliary calls.
    "default_aux_model": "anyrouter/hermes",
    # Opt-in app-attribution headers; empty dict when not enabled.
    "default_headers": _default_headers(),
    # Shown only when the live catalog is unreachable. Flagship tool-capable
    # routes spanning the vendors AnyRouter fronts; verified against the live
    # catalog.
    "fallback_models": (
        "anyrouter/hermes",
        "anyrouter/auto",
        "x-ai/grok-4.7",
        "moonshotai/kimi-k3",
        "deepseek/deepseek-v4-flash",
        "google/gemini-3.5-flash",
        "z-ai/glm-5.3-flash",
        "minimax/m3",
    ),
}

anyrouter = AnyRouterProfile(**_supported_kwargs(AnyRouterProfile, PROFILE_FIELDS))

register_provider(anyrouter)
