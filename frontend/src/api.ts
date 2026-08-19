/**
 * Typed client for the JARVIS FastAPI backend.
 *
 * Every network call funnels through `request()` so error handling, JSON
 * decoding and the base URL are defined exactly once.
 */

import type {
  BriefingHistoryItem,
  BriefingResponse,
  ChatResponse,
  Preferences,
  ServerStatus,
} from './types';

/**
 * Backend origin. Override per environment with `VITE_API_BASE` in
 * `frontend/.env`; the default matches `API_PORT` in the root `.env`.
 */
export const API_BASE: string =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, '') ?? 'http://localhost:8000';

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
    ...init,
  });

  if (!response.ok) {
    // FastAPI reports failures as `{ "detail": ... }`; fall back to the status text.
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* response had no JSON body — keep the status text */
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

/** Capability flags and engine metadata. Also serves as the reachability probe. */
export const getStatus = () => request<ServerStatus>('/');

export const getPreferences = () => request<Preferences>('/preferences');

export const savePreferences = (preferences: Preferences) =>
  request<Preferences>('/preferences', {
    method: 'PUT',
    body: JSON.stringify(preferences),
  });

export const getHistory = (limit = 10) =>
  request<{ history: BriefingHistoryItem[] }>(`/history?limit=${limit}`);

/** Runs the full six-agent workflow. Expect this to take tens of seconds. */
export const runBriefing = () => request<BriefingResponse>('/briefing', { method: 'POST' });

export const sendChatMessage = (message: string) =>
  request<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({ message }),
  });
