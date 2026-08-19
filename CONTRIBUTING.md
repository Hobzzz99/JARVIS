# Contributing

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # then fill in your keys (optional — see below)

cd frontend && npm install && cd ..
```

Keys are optional. With none set, JARVIS runs fully offline on local
HuggingFace models and serves labelled sample articles.

## Checks

Run these before opening a pull request — CI runs the same commands.

```bash
ruff check .              # lint
ruff format .             # format
pytest                    # backend tests
pytest --cov              # with coverage

cd frontend
npm run lint
npm run build             # type-checks via `tsc -b`
```

## Conventions

**Backend**

- Every module gets a docstring explaining *why* it exists, not just what it is.
- Read configuration through `config.get_settings()` — never `os.getenv` inline.
- Log through `config.get_logger(__name__)` — never `print` outside the
  delivery agent's terminal banner.
- Agents must degrade, not crash: if a model or API is unavailable, log a
  warning and fall back. A briefing should always be produced.
- Public functions carry type hints and a Google-style docstring.

**Frontend**

- Types mirroring API responses live in `src/types.ts` and must stay in sync
  with the Pydantic models in `api/main.py`.
- All network calls go through `src/api.ts`. No bare `fetch` in components.
- Static configuration belongs in `src/constants.ts`.

## Tests

Tests are hermetic: no API keys, no network calls, no model downloads. The
`isolated_env` fixture sets `JARVIS_SKIP_DOTENV=1` and redirects memory to a
temp directory, so a developer's real `.env` and briefing archive are never
touched.

When adding a fallback path, add a test that forces the primary path to fail
and asserts the fallback produced usable output — that behaviour is the point
of the architecture.

## Security

Never commit `.env`, API keys, or tokens. `.gitignore` blocks `.env` by
default; if you add a new secret-bearing file, add it there too.
