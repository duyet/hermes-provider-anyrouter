AnyRouter provider installed.

1. Add your key to `~/.hermes/.env`:

   ```
   ANYROUTER_API_KEY=sk-...
   ```

   Get one at https://anyrouter.dev — the same key also works against the
   standard `https://anyrouter.dev/api/v1` OpenAI-compatible endpoint.

2. Pick a model: run `hermes model` and choose `anyrouter`, or launch with
   `hermes --provider anyrouter -m anyrouter/hermes`.

   `anyrouter/hermes` is a meta-model that auto-routes across the top
   tool-calling models with fallback — a good default for agentic work.

3. Optional:

   - `ANYROUTER_BASE_URL` — point at a proxy or self-hosted relay instead of
     the default endpoint.
   - `ANYROUTER_APP_ATTRIBUTION=1` — opt in to AnyRouter's app-attribution
     headers so Hermes traffic is credited in the public app rankings.
