# hermes-provider-anyrouter

[AnyRouter](https://anyrouter.dev) model-provider plugin for
[Hermes Agent](https://github.com/NousResearch/hermes-agent).

It registers AnyRouter as the `anyrouter` provider over its OpenAI-compatible
Chat Completions endpoint (`https://anyrouter.dev/api/v1`), and keeps the
`/model` picker filtered to routes that can actually call tools. It touches
no Hermes core files — discovery, credential resolution, `hermes doctor`, and
the `--provider` flag all auto-wire from the provider registry.

Part of the Hermes plugin ecosystem: install it from the plugin catalog, from
git, or via pip — no core files required.

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
- **Privacy-respecting by default.** App-attribution headers are opt-in only
  (`ANYROUTER_APP_ATTRIBUTION=1`) — nothing is tagged unless you ask for it.

## Requirements

- Hermes Agent — any build with the model-provider plugin system
  (`providers.register_provider` + `providers.base.ProviderProfile`).
  Verified on 0.21.5.
- An AnyRouter API key — get one at <https://anyrouter.dev>.

## Install

Pick one of three install paths:

```bash
# 1. From the Hermes plugin catalog:
hermes plugins install anyrouter

# 2. Straight from git:
hermes plugins install https://github.com/duyet/hermes-provider-anyrouter --subdir anyrouter

# 3. As a pip package (picked up via the hermes_agent.plugins entry point):
pip install git+https://github.com/duyet/hermes-provider-anyrouter.git
```

Verify it's registered:

```bash
hermes plugins list          # should show `anyrouter`, enabled
hermes doctor                # provider health check
```

## Configure

Add your API key to `~/.hermes/.env`:

```
ANYROUTER_API_KEY=sk-ar-...
```

| Variable | Required | Purpose |
|---|---|---|
| `ANYROUTER_API_KEY` | yes | AnyRouter API key |
| `ANYROUTER_BASE_URL` | no | Override the default endpoint (proxy / self-hosted relay) |
| `ANYROUTER_APP_ATTRIBUTION` | no | `1` opts in to app-attribution headers crediting Hermes traffic in AnyRouter's public app rankings |

## Use

```bash
hermes model                                   # interactive picker → anyrouter
hermes --provider anyrouter -m anyrouter/hermes
```

Or set it as your default in `~/.hermes/config.yaml`:

```yaml
model:
  provider: anyrouter
  default: anyrouter/hermes
```

### Recommended models

| Model | Notes |
|---|---|
| `anyrouter/hermes` | Meta-model auto-routing across top tool-calling models, with fallback — a good default for agentic work |
| `anyrouter/auto` | Auto-routes across coding/vision-capable models |
| `x-ai/grok-4.7` | Direct vendor route |
| `moonshotai/kimi-k3` | 1M context, agentic |
| `deepseek/deepseek-v4-flash` | Fast, reasoning-capable |

Run `hermes model` to see the live filtered list — it's read fresh from the
AnyRouter catalog, so new tool-capable routes appear automatically.

## Troubleshooting

| Symptom | Check |
|---|---|
| `unknown provider: anyrouter` | `hermes plugins list` — is the plugin enabled? New sessions pick it up after enable. |
| `Model 'x' not found` | Run `hermes model` — the picker shows only routes AnyRouter currently serves with tool-calling support. |
| Catalog unreachable | The picker falls back to a built-in model list; check your network / `ANYROUTER_BASE_URL`. |
| Upstream 429/402 errors | AnyRouter's upstreams are rate-limited or out of credits — retry later or pick a different model; check `GET /api/v1/models`. |

Logs live under `~/.hermes/logs/` — `hermes logs --follow` is the quickest
way to watch a request go through.

## Hermes version

There is no version pin. Every contact point with core
(`_supported_kwargs`, `_inherited_fetch`, `hermes_cli.urllib_security`) is
resolved by introspection, so newer profile fields are used when present and
skipped on older builds.

## Development

```bash
python -m unittest discover -s tests -v
```

The tests run fully offline: `tests/conftest_stub.py` stubs the Hermes
`providers` runtime, including a legacy profile base, so both modern and
older Hermes builds are exercised.

## Links

- AnyRouter docs: <https://docs.anyrouter.dev>
- AnyRouter MCP server (`list_models`, `get_credits`, key management inside
  your agent): `https://anyrouter.dev/api/v1/mcp` — add it under
  `mcp_servers` in `~/.hermes/config.yaml`

## License

MIT — see [LICENSE](LICENSE).
