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
from decimal import Decimal, InvalidOperation
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

# First-party meta-models carry their own rate in the catalog. Upstream slugs
# served through AnyRouter are left to Hermes' generic pricing path.
_PRICED_PREFIX = "anyrouter/"
_ONE_MILLION = Decimal("1000000")


@dataclasses.dataclass(frozen=True)
class _Rate:
    """Per-1M USD rates parsed from one catalog entry's ``pricing`` block."""

    input_per_1m: Decimal
    output_per_1m: Decimal
    byok: bool


@dataclasses.dataclass(frozen=True)
class _Catalog:
    """One fetched ``/v1/models`` listing: picker ids plus per-model rates."""

    models: list[str]
    pricing: dict[str, _Rate]


# Keyed by API key (``""`` = anonymous). An authenticated list also carries the
# caller's own ``@preset/<slug>`` entries, so one key's list must never be
# served to another, nor an anonymous list to a keyed caller.
_CATALOG_CACHE: dict[str, _Catalog] = {}


def _agentic(entry: dict[str, Any]) -> bool:
    """True when a catalog entry advertises tool-calling support."""
    caps = entry.get("capabilities")
    if isinstance(caps, list) and _AGENTIC_CAPABILITIES & set(caps):
        return True
    # ``supported_parameters`` is the authoritative OpenAI-style signal; a route
    # that accepts ``tools`` is usable by an agent even if the tag is missing.
    params = entry.get("supported_parameters")
    return isinstance(params, list) and "tools" in params


def _parse_rate(pricing: Any) -> _Rate | None:
    """Usable rates from a catalog ``pricing`` block, or None."""
    if not isinstance(pricing, dict):
        return None
    input_rate = _per_million(pricing, "input_per_1m")
    output_rate = _per_million(pricing, "output_per_1m")
    if input_rate is None or output_rate is None:
        return None
    return _Rate(input_rate, output_rate, byok=pricing.get("billing") == "byok")


def _per_million(pricing: dict[str, Any], key: str) -> Decimal | None:
    """Read a non-negative per-1M USD rate from a catalog pricing block."""
    raw = pricing.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        rate = Decimal(str(raw))
    except InvalidOperation:
        return None
    return rate if rate.is_finite() and rate >= 0 else None


def _cached_rate(model: str) -> _Rate | None:
    """Rate for *model* from any cached catalog, without network I/O.

    The cost hook is not told which API key served the request. A first-party
    ``anyrouter/*`` rate is the same in every listing, so any cached catalog
    that carries the model answers it. Iterates a snapshot: another thread
    may cache a catalog for a different key mid-lookup.
    """
    for catalog in tuple(_CATALOG_CACHE.values()):
        rate = catalog.pricing.get(model)
        if rate is not None:
            return rate
    return None


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
        caller_base = (base_url or "").strip()
        custom_base = bool(caller_base) and (
            caller_base.rstrip("/") != self.base_url.rstrip("/")
        )
        if custom_base:
            return self._inherited_fetch(
                api_key=api_key, base_url=base_url, timeout=timeout
            )

        cache_key = api_key or ""
        cached = _CATALOG_CACHE.get(cache_key)
        if cached is not None:
            return cached.models

        catalog = self._fetch_catalog(api_key=api_key, timeout=timeout)
        if catalog is not None:
            _CATALOG_CACHE[cache_key] = catalog
            return catalog.models

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

    def _fetch_catalog(
        self, *, api_key: str | None, timeout: float
    ) -> _Catalog | None:
        """Fetch the catalog: tool-capable ids plus every entry's pricing."""
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
        pricing: dict[str, _Rate] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("model_id") or entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            rate = _parse_rate(entry.get("pricing"))
            if rate is not None:
                pricing.setdefault(model_id, rate)
            if _agentic(entry) and model_id not in seen:
                seen.add(model_id)
                models.append(model_id)

        if not models:
            logger.debug("fetch_models(anyrouter): catalog filtered to zero models")
            return None
        return _Catalog(models=models, pricing=pricing)

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """Estimate a request's cost from the cached AnyRouter catalog rate.

        Only first-party ``anyrouter/*`` meta-models are priced here; upstream
        slugs fall through to Hermes' generic pricing. The hot path never
        touches the network — a model missing from the cached catalog (or a
        Hermes build without ``CostResult``) yields None and the generic path.

        The catalog lists one input rate and no cache rate, so cached prompt
        tokens are charged at the input rate. Meta-models bill each hop at the
        underlying model's rate, so the result is an estimate, not an invoice.
        """
        if not model.startswith(_PRICED_PREFIX):
            return None
        rate = _cached_rate(model)
        if rate is None:
            logger.debug(
                "get_usage_cost(anyrouter): no usable cached rate for %s", model
            )
            return None
        try:
            from agent.usage_pricing import CostResult, format_cost_label
        except ImportError:
            logger.debug("get_usage_cost(anyrouter): this Hermes build has no CostResult")
            return None

        try:
            prompt_tokens = int(usage.prompt_tokens)
            output_tokens = int(usage.output_tokens)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("get_usage_cost(anyrouter): unreadable usage: %s", exc)
            return None

        amount = (
            Decimal(prompt_tokens) * rate.input_per_1m
            + Decimal(output_tokens) * rate.output_per_1m
        ) / _ONE_MILLION
        notes = ["AnyRouter catalog rate; meta-models bill each hop at its own rate."]
        if rate.byok:
            notes.append("BYOK route: billed by your upstream provider, not AnyRouter.")
        return CostResult(
            amount_usd=amount,
            status="estimated",
            source="provider_models_api",
            label=format_cost_label(amount),
            notes=tuple(notes),
        )

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
