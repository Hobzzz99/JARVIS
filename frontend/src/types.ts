/**
 * Shared types mirroring the FastAPI response schemas in `api/main.py`.
 * Keep these in sync with the Pydantic models on the backend.
 */

export interface Preferences {
  name: string;
  interests: string[];
  favorite_sources: string[];
}

export interface Article {
  title: string;
  description: string;
  url: string;
  source: string;
  published: string;
  /** Zero-shot classifier confidence in [0, 1]; absent when ranking is disabled. */
  relevance_score?: number;
  /** Focus topic the article matched most strongly. */
  top_topic?: string;
  /** True for the offline demo payload served when NEWS_API_KEY is unset. */
  is_sample?: boolean;
}

export interface Paper {
  title: string;
  summary: string;
  url: string;
  authors: string[];
  published: string;
}

export interface BriefingHistoryItem {
  date: string;
  briefing: string;
  created_at?: string;
}

/** Per-node timing emitted by the LangGraph workflow. */
export interface TelemetryEntry {
  node: string;
  seconds: number;
  status: string;
}

export interface BriefingResponse {
  briefing: string;
  articles: Article[];
  papers: Paper[];
  insights: string;
  focus_topics: string[];
  telemetry: TelemetryEntry[];
  duration_seconds: number;
}

/** Capability flags from `GET /` — drives the dashboard's status banners. */
export interface ServerStatus {
  status: string;
  llm: string;
  gemini_enabled: boolean;
  news_api_enabled: boolean;
  hf_ranking: boolean;
  hf_summarizer: boolean;
  offline_mode: boolean;
}

export interface ChatResponse {
  response: string;
  source: 'gemini' | 'fallback';
}

export type TabId = 'briefing' | 'history' | 'preferences';

/* -------------------------------------------------------------------------
 * Web Speech API
 * The DOM lib ships no types for SpeechRecognition (it is still a draft spec
 * behind a `webkit` prefix), so these cover only the surface we consume.
 * ---------------------------------------------------------------------- */

export interface SpeechRecognitionResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}

export interface SpeechRecognitionErrorEvent {
  error: string;
}

export interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onresult: ((event: SpeechRecognitionResultEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  start(): void;
  stop(): void;
  abort?(): void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}
