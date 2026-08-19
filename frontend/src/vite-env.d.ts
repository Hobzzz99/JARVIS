/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Origin of the JARVIS FastAPI backend. See `frontend/.env.example`. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
