# hermes-provider-anyrouter

[AnyRouter](https://anyrouter.dev) model-provider plugin for
[Hermes Agent](https://github.com/NousResearch/hermes-agent).

It registers AnyRouter as the `anyrouter` provider over its OpenAI-compatible
Chat Completions endpoint (`https://anyrouter.dev/api/v1`), and keeps the
`/model` picker filtered to routes that can actually call tools. It touches
no Hermes core files — discovery, credential resolution, `hermes doctor`, and
the `--provider` flag all auto-wire from the provider registry.

App-attribution headers are **opt-in** (`ANYROUTER_APP_ATTRIBUTION=1`) and off
by default — nothing is tagged unless you ask for it.

## Why this lives outside the Hermes tree

Hermes closes PRs that add third-party product integrations under `plugins/`
in the main repo — a coupling-and-maintenance decision, not a quality bar
("No new third-party-product plugins in-tree"; see the contributor instructions
in the Hermes repository). Standalone plugin repos are the supported path.
This plugin exists because
[NousResearch/hermes-agent#54714](https://github.com/NousResearch/hermes-agent/pull/54714)
was closed under that policy, and is submitted to the
[plugin catalog](https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog)
as a catalog entry instead.

## What it does

- **Tool-capable picker filtering.** AnyRouter's public `/api/v1/models`
  catalog annotates each route with a `capabilities` list; the plugin keeps
  only routes advertising `function-calling`/`tools`, so the picker never
  offers a model the agent can't drive. Falls back to the generic `/v1/models`
  path if the catalog is unreachable or a custom `ANYROUTER_BASE_URL` points
  at a relay whose catalog isn't ours to interpret.
- **Request shaping** matching AnyRouter's documented API contract:
  - `extra_body.session_id` — sticky-session grouping in the Request Logs
    dashboard.
  - `extra_body.provider` — request-level routing preferences
    (`only` / `ignore` / `order` / `sort` / `allow_fallbacks` / `max_price`).
  - `extra_body.reasoning` — reasoning effort passthrough; AnyRouter
    translates it to each upstream's native thinking control.
- **First-party meta-models** like `anyrouter/hermes` (auto-routes across top
  tool-calling models with fallback) are surfaced alongside `vendor/model`
  routes like `x-ai/grok-4.7` and `moonshotai/kimi-k3`.

## Install

```bash
# From the Hermes plugin catalog (once the entry lands):
hermes plugins install anyrouter

# Or straight from git:
hermes plugins install https://github.com/duyet/hermes-provider-anyrouter --subdir anyrouter

# Or as a pip plugin (picked up via the hermes_agent.plugins entry point):
pip install git+https://github.com/duyet/hermes-provider-anyrouter.git
```

Then add your key to `~/.hermes/.env`:

```
ANYROUTER_API_KEY=sk-...
```

and pick the provider:

```bash
hermes model                          # interactive picker
hermes --provider anyrouter -m anyrouter/hermes
```

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `ANYROUTER_API_KEY` | yes | AnyRouter API key |
| `ANYROUTER_BASE_URL` | no | Override the default endpoint (proxy / self-hosted relay) |
| `ANYROUTER_APP_ATTRIBUTION` | no | `1` opts in to app-attribution headers crediting Hermes traffic in AnyRouter's public app rankings |

## Hermes version

The plugin only needs `providers.register_provider` and
`providers.base.ProviderProfile`. There is no version pin: every contact point
with core (`_supported_kwargs`, `_inherited_fetch`, `hermes_cli.urllib_security`)
is resolved by introspection, so newer profile fields are used when present
and skipped on older builds.

Verified on 0.21.5.

## Development

```bash
python -m unittest discover -s tests -v
```

The tests run fully offline: `tests/conftest_stub.py` stubs the Hermes
`providers` runtime, including a legacy profile base, so both modern and
older Hermes builds are exercised.

## License

MIT — see [LICENSE](LICENSE).
