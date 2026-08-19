# JARVIS Dashboard

React 19 + TypeScript + Tailwind v4 control panel for the JARVIS briefing
pipeline. Built with Vite.

## Running

```bash
npm install
npm run dev          # http://localhost:5173
```

The dashboard needs the backend running. From the repository root:

```bash
uvicorn api.main:app --reload --port 8000
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE` | `http://localhost:8000` | Origin of the FastAPI backend |

Copy `.env.example` to `.env` to override. The value must match `API_PORT` in
the root `.env`, and the dashboard's origin must appear in the backend's
`CORS_ORIGINS`.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Dev server with HMR |
| `npm run build` | Type-check (`tsc -b`) then production build |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint |

## Layout

```
src/
├── App.tsx        Dashboard component: HUD, feed, voice I/O, tabs
├── api.ts         Typed client — every backend call goes through here
├── types.ts       Response types mirroring the Pydantic models in api/main.py
├── constants.ts   Categories, keyword filters, tab and voice configuration
├── index.css      Tailwind layer + the HUD/arc-reactor visual system
└── App.css        Component-scoped styles
```

`types.ts` must stay in sync with `api/main.py`. If you change a response
model on the backend, update it here in the same commit — `npm run build`
type-checks the consumers and will catch the mismatch.

## Voice interface

Speech recognition uses the Web Speech API, which is still vendor-prefixed and
unavailable in Firefox. Support is detected once at mount; when absent the mic
control is disabled and a notice points to the text input. Speech synthesis
prefers a UK English male voice, falling back to any `en-GB` voice installed.

Spoken commands:

| Phrase contains | Action |
|---|---|
| "briefing", "compile", "run brief" | Runs the full workflow |
| "mute", "silence", "stop talking" | Disables speech output |
| "unmute", "talk to me" | Re-enables speech output |
| anything else | Routed to `POST /chat` |
