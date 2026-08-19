/** Static UI configuration extracted from the dashboard component. */

import type { TabId } from './types';

/** Sidebar filters. Purely client-side — applied to whatever the last run returned. */
export const CATEGORIES = [
  'All',
  'Research',
  'Companies',
  'Breakthroughs',
  'Models',
  'Robotics',
  'Ethics',
  'Startups',
] as const;

export type Category = (typeof CATEGORIES)[number];

/**
 * Keyword sets backing the category filter. Matching is a case-insensitive
 * substring test against each item's title and description.
 */
export const CATEGORY_KEYWORDS: Record<string, string[]> = {
  Research: ['research', 'paper', 'arxiv', 'study', 'university', 'scientific', 'preprint'],
  Companies: [
    'google',
    'meta',
    'openai',
    'microsoft',
    'anthropic',
    'nvidia',
    'company',
    'corp',
    'deepmind',
  ],
  Breakthroughs: ['breakthrough', 'milestone', 'revolution', 'new standard', 'unveils', 'pioneers'],
  Models: ['gpt', 'gemini', 'claude', 'llama', 'model', 'weights', 'mistral', 'bart'],
  Robotics: ['robot', 'humanoid', 'computer use', 'device', 'physical', 'hardware', 'actuator'],
  Ethics: ['ethics', 'safety', 'risk', 'policy', 'law', 'govern', 'regulate', 'guardrail'],
  Startups: ['startup', 'yc', 'raise', 'fund', 'venture', 'launches', 'inc'],
};

export const TABS: { id: TabId; label: string }[] = [
  { id: 'briefing', label: 'Command Feed' },
  { id: 'history', label: 'Archived Logs' },
  { id: 'preferences', label: 'Preferences' },
];

/** Poll interval for the backend reachability check. */
export const SERVER_POLL_MS = 10_000;

/** Tick rate for the HUD clock and simulated telemetry gauges. */
export const HUD_TICK_MS = 1_000;

/**
 * Voice names preferred for the JARVIS persona, in priority order. The browser
 * picks whichever is installed; `en-GB` is the fallback locale.
 */
export const PREFERRED_VOICES = ['Google UK English Male', 'Microsoft David', 'Daniel'];
