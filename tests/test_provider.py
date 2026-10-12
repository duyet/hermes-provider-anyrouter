"""Offline tests for the AnyRouter provider profile."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.request
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest_stub  # noqa: E402


def _catalog_payload(models):
    return json.dumps({"object": "list", "data": models}).encode()


def _entry(model_id, capabilities=None, pricing=None):
    entry = {
        "id": model_id,
        "object": "model",
        "capabilities": capabilities if capabilities is not None else ["chat", "function-calling"],
    }
    if pricing is not None:
        entry["pricing"] = pricing
    return entry


def _rate(input_per_1m, output_per_1m, **extra):
    return {
        "prompt": str(input_per_1m),
        "completion": str(output_per_1m),
        "input_per_1m": input_per_1m,
        "output_per_1m": output_per_1m,
        **extra,
    }


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRegistration(unittest.TestCase):
    def test_registers_as_anyrouter(self):
        _, profile = conftest_stub.load_plugin()
        self.assertEqual(profile.name, "anyrouter")

    def test_profile_fields(self):
        _, profile = conftest_stub.load_plugin()
        self.assertEqual(profile.base_url, "https://anyrouter.dev/api/v1")
        self.assertEqual(profile.models_url, "https://anyrouter.dev/api/v1/models")
        self.assertEqual(profile.env_vars, ("ANYROUTER_API_KEY", "ANYROUTER_BASE_URL"))
        self.assertEqual(profile.auth_type, "api_key")
        self.assertEqual(profile.api_mode, "chat_completions")
        self.assertEqual(profile.hostname, "anyrouter.dev")

    def test_fallback_models_all_vendor_slugs(self):
        _, profile = conftest_stub.load_plugin()
        self.assertTrue(profile.fallback_models)
        for m in profile.fallback_models:
            self.assertIn("/", m)

    def test_importable_on_legacy_profile_base(self):
        # Older builds lack supports_vision/hostname/default_aux_model; the
        # plugin must still register rather than TypeError at import.
        _, profile = conftest_stub.load_plugin(legacy=True)
        self.assertEqual(profile.name, "anyrouter")
        self.assertEqual(profile.base_url, "https://anyrouter.dev/api/v1")


class TestAttributionOptIn(unittest.TestCase):
    def test_headers_off_by_default(self):
        env = dict(os.environ)
        env.pop("ANYROUTER_APP_ATTRIBUTION", None)
        with patch.dict(os.environ, env, clear=True):
            _, profile = conftest_stub.load_plugin()
        self.assertEqual(profile.default_headers, {})

    def test_headers_on_when_opted_in(self):
        with patch.dict(os.environ, {"ANYROUTER_APP_ATTRIBUTION": "1"}):
            _, profile = conftest_stub.load_plugin()
        self.assertEqual(
            profile.default_headers.get("X-AnyRouter-Title"), "Hermes Agent"
        )
        self.assertIn("HTTP-Referer", profile.default_headers)


class TestExtraBody(unittest.TestCase):
    def setUp(self):
        _, self.profile = conftest_stub.load_plugin()

    def test_session_id_passthrough(self):
        body = self.profile.build_extra_body(session_id="sess-1")
        self.assertEqual(body["session_id"], "sess-1")

    def test_provider_preferences_passthrough(self):
        prefs = {"order": ["DeepInfra"], "allow_fallbacks": True}
        body = self.profile.build_extra_body(provider_preferences=prefs)
        self.assertEqual(body["provider"], prefs)

    def test_no_session_no_prefs(self):
        self.assertEqual(self.profile.build_extra_body(), {})


class TestReasoningPassthrough(unittest.TestCase):
    def setUp(self):
        _, self.profile = conftest_stub.load_plugin()

    def test_reasoning_passthrough(self):
        cfg = {"effort": "high"}
        extra, top = self.profile.build_api_kwargs_extras(
            reasoning_config=cfg, supports_reasoning=True
        )
        self.assertEqual(extra["reasoning"], cfg)
        self.assertEqual(top, {})

    def test_reasoning_default_when_enabled_without_config(self):
        extra, _ = self.profile.build_api_kwargs_extras(
            reasoning_config=None, supports_reasoning=True
        )
        self.assertEqual(
            extra["reasoning"], {"enabled": True, "effort": "medium"}
        )

    def test_reasoning_omitted_when_unsupported(self):
        extra, top = self.profile.build_api_kwargs_extras(
            reasoning_config={"effort": "high"}, supports_reasoning=False
        )
        self.assertEqual(extra, {})
        self.assertEqual(top, {})


class TestFetchModels(unittest.TestCase):
    def setUp(self):
        self.module, self.profile = conftest_stub.load_plugin()
        self.module._CATALOG_CACHE = {}

    def _stub_open(self, payload):
        resp = _FakeResponse(payload)
        return lambda req, timeout=8.0: resp

    def _patch_opener(self, payload):
        mod = sys.modules["hermes_cli.urllib_security"]
        return patch.object(mod, "open_credentialed_url", self._stub_open(payload))

    def test_filters_to_tool_capable(self):
        payload = _catalog_payload([
            _entry("anyrouter/hermes"),
            _entry("nvidia/nemotron-3.5-content-safety", ["chat", "vision"]),
            _entry("x-ai/grok-4.7", ["chat", "function-calling", "vision"]),
            _entry("nvidia/nemotron-3-embed-1b", ["embedding"]),
            _entry("z-ai/glm-5.3-flash", ["chat", "agentic", "tools"]),
        ])
        with self._patch_opener(payload):
            models = self.profile.fetch_models(api_key="k")
        self.assertEqual(
            models, ["anyrouter/hermes", "x-ai/grok-4.7", "z-ai/glm-5.3-flash"]
        )

    def test_keeps_models_that_accept_tools_without_the_tag(self):
        entry = _entry("nvidia/nemotron-3-super-120b-a12b", ["chat"])
        entry["supported_parameters"] = ["temperature", "tools"]
        with self._patch_opener(_catalog_payload([entry, _entry("a/b", ["chat"])])):
            models = self.profile.fetch_models(api_key="k")
        self.assertEqual(models, ["nvidia/nemotron-3-super-120b-a12b"])

    def test_cache_is_per_api_key_so_presets_do_not_leak(self):
        mine = _catalog_payload([_entry("a/b"), _entry("@preset/mine")])
        other = _catalog_payload([_entry("a/b")])
        with self._patch_opener(mine):
            self.assertIn("@preset/mine", self.profile.fetch_models(api_key="k1"))
        with self._patch_opener(other):
            self.assertNotIn("@preset/mine", self.profile.fetch_models(api_key="k2"))

    def test_dedupes(self):
        payload = _catalog_payload([
            _entry("a/b"), _entry("a/b"), _entry("c/d"),
        ])
        with self._patch_opener(payload):
            models = self.profile.fetch_models()
        self.assertEqual(models, ["a/b", "c/d"])

    def test_falls_back_when_catalog_empty(self):
        with self._patch_opener(_catalog_payload([])):
            models = self.profile.fetch_models()
        self.assertEqual(models, conftest_stub.SUPER_MARKER)

    def test_falls_back_when_all_filtered(self):
        payload = _catalog_payload([
            _entry("m/e", ["embedding"]), _entry("m/t", ["chat"]),
        ])
        with self._patch_opener(payload):
            models = self.profile.fetch_models()
        self.assertEqual(models, conftest_stub.SUPER_MARKER)

    def test_falls_back_on_bad_payload(self):
        with self._patch_opener(b"not json"):
            models = self.profile.fetch_models()
        self.assertEqual(models, conftest_stub.SUPER_MARKER)

    def test_falls_back_on_transport_error(self):
        def boom(req, timeout=8.0):
            raise OSError("no route")

        mod = sys.modules["hermes_cli.urllib_security"]
        with patch.object(mod, "open_credentialed_url", boom):
            models = self.profile.fetch_models()
        self.assertEqual(models, conftest_stub.SUPER_MARKER)

    def test_custom_base_url_skips_catalog(self):
        # A proxy/relay catalog is not ours to interpret — generic path only.
        with self._patch_opener(_catalog_payload([_entry("a/b")])):
            models = self.profile.fetch_models(
                base_url="https://relay.internal/v1"
            )
        self.assertEqual(models, conftest_stub.SUPER_MARKER)

    def test_legacy_base_signature_still_falls_back(self):
        module, profile = conftest_stub.load_plugin(legacy=True)
        module._CATALOG_CACHE = {}

        def boom(req, timeout=8.0):
            raise OSError("down")

        mod = sys.modules["hermes_cli.urllib_security"]
        with patch.object(mod, "open_credentialed_url", boom):
            models = profile.fetch_models()
        # Legacy base has no base_url kwarg; must not TypeError on fallback.
        self.assertEqual(models, conftest_stub.LEGACY_SUPER_MARKER)


class TestUsageCost(unittest.TestCase):
    def setUp(self):
        self.module, self.profile = conftest_stub.load_plugin()
        self.module._CATALOG_CACHE = {}
        self.usage = conftest_stub.StubCanonicalUsage(
            input_tokens=800_000, cache_read_tokens=200_000, output_tokens=500_000
        )

    def _prime(self, entries, api_key="k"):
        resp = _FakeResponse(_catalog_payload(entries))
        mod = sys.modules["hermes_cli.urllib_security"]
        with patch.object(mod, "open_credentialed_url", lambda req, timeout=8.0: resp):
            self.profile.fetch_models(api_key=api_key)

    def _no_network(self):
        def boom(req, timeout=8.0):
            raise AssertionError("get_usage_cost must not hit the network")

        mod = sys.modules["hermes_cli.urllib_security"]
        return patch.object(mod, "open_credentialed_url", boom)

    def test_prices_meta_model_from_cached_catalog(self):
        self._prime([_entry("anyrouter/hermes", pricing=_rate(0.5, 2))])
        with self._no_network():
            cost = self.profile.get_usage_cost("anyrouter/hermes", self.usage)
        # (800k input + 200k cache-read) * $0.5/M + 500k * $2/M = $0.50 + $1.00
        self.assertEqual(cost.amount_usd, Decimal("1.5"))
        self.assertEqual(cost.status, "estimated")
        self.assertEqual(cost.source, "provider_models_api")
        self.assertEqual(cost.label, "~$1.5000")

    def test_catalog_cache_keeps_ids_and_pricing(self):
        self._prime([
            _entry("anyrouter/hermes", pricing=_rate(0.5, 2)),
            _entry("nvidia/nemotron-3-embed-1b", ["embedding"], pricing=_rate(0, 0)),
        ])
        catalog = self.module._CATALOG_CACHE["k"]
        self.assertEqual(catalog.models, ["anyrouter/hermes"])
        # Pricing covers every entry, not only the picker's tool-capable ones.
        self.assertIn("nvidia/nemotron-3-embed-1b", catalog.pricing)
        with self._no_network():
            self.assertEqual(self.profile.fetch_models(api_key="k"), ["anyrouter/hermes"])

    def test_byok_route_is_noted(self):
        self._prime([
            _entry("anyrouter/decision", pricing=_rate(0.04, 0, billing="byok")),
        ])
        cost = self.profile.get_usage_cost("anyrouter/decision", self.usage)
        self.assertEqual(cost.amount_usd, Decimal("0.04"))
        self.assertTrue(any("BYOK" in n for n in cost.notes))

    def test_upstream_slug_defers_to_base(self):
        self._prime([_entry("x-ai/grok-4.7", pricing=_rate(1.6, 4.8))])
        self.assertIsNone(self.profile.get_usage_cost("x-ai/grok-4.7", self.usage))

    def test_cache_miss_returns_none_without_network(self):
        with self._no_network():
            self.assertIsNone(self.profile.get_usage_cost("anyrouter/hermes", self.usage))

    def test_missing_or_bad_rate_returns_none(self):
        self._prime([
            _entry("anyrouter/a", pricing={"prompt": "0.5"}),
            _entry("anyrouter/b", pricing=_rate("free", 1)),
            _entry("anyrouter/c", pricing=_rate(-1, 1)),
            _entry("anyrouter/d", pricing=_rate(True, 1)),
        ])
        for model in ("anyrouter/a", "anyrouter/b", "anyrouter/c", "anyrouter/d"):
            self.assertIsNone(self.profile.get_usage_cost(model, self.usage), model)

    def test_unreadable_usage_returns_none(self):
        self._prime([_entry("anyrouter/hermes", pricing=_rate(0.5, 2))])
        self.assertIsNone(self.profile.get_usage_cost("anyrouter/hermes", object()))

    def test_legacy_build_without_cost_result_returns_none(self):
        module, profile = conftest_stub.load_plugin(legacy=True)
        module._CATALOG_CACHE = {}
        resp = _FakeResponse(_catalog_payload([
            _entry("anyrouter/hermes", pricing=_rate(0.5, 2)),
        ]))
        mod = sys.modules["hermes_cli.urllib_security"]
        with patch.object(mod, "open_credentialed_url", lambda req, timeout=8.0: resp):
            profile.fetch_models(api_key="k")
        self.assertIsNone(profile.get_usage_cost("anyrouter/hermes", self.usage))


if __name__ == "__main__":
    unittest.main()
