import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Settings,
  History,
  Cpu,
  User,
  Plus,
  X,
  ChevronRight,
  BookOpen,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Compass,
  Activity,
  Shield,
  Clock,
  FileText,
  Search,
  TrendingUp,
  Brain,
  Filter,
} from 'lucide-react';

import * as api from './api';
import {
  CATEGORIES,
  CATEGORY_KEYWORDS,
  HUD_TICK_MS,
  PREFERRED_VOICES,
  SERVER_POLL_MS,
  TABS,
} from './constants';
import type {
  Article,
  BriefingHistoryItem,
  Paper,
  Preferences,
  ServerStatus,
  SpeechRecognitionLike,
  TabId,
  TelemetryEntry,
} from './types';

const DEFAULT_PREFERENCES: Preferences = {
  name: 'Operator',
  interests: [],
  favorite_sources: [],
};

/** Narrow an unknown thrown value to a displayable message. */
const describeError = (error: unknown, fallback: string): string =>
  error instanceof Error && error.message ? error.message : fallback;

export default function App() {
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
  const [history, setHistory] = useState<BriefingHistoryItem[]>([]);
  const [briefing, setBriefing] = useState<string | null>(null);
  const [articles, setArticles] = useState<Article[]>([]);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryEntry[]>([]);
  const [lastRunSeconds, setLastRunSeconds] = useState<number | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Feed filtering
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [askJarvisQuery, setAskJarvisQuery] = useState<string>('');
  const [globalFilter, setGlobalFilter] = useState<string>('Worldwide');

  // HUD gauges — decorative, driven by a local tick rather than host metrics.
  const [coreTemp, setCoreTemp] = useState(31.8);
  const [memoryUsage, setMemoryUsage] = useState(44.5);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());

  // Voice I/O
  const [isListening, setIsListening] = useState(false);
  const [voiceActive, setVoiceActive] = useState(true);
  // Evaluated once at mount — the Web Speech API is still vendor-prefixed and
  // absent entirely in Firefox, so the mic control is disabled rather than dead.
  const [speechSupported] = useState(
    () => typeof window !== 'undefined' && Boolean(window.SpeechRecognition ?? window.webkitSpeechRecognition),
  );
  const [isJarvisSpeaking, setIsJarvisSpeaking] = useState(false);
  const [userTranscript, setUserTranscript] = useState<string | null>(null);
  const [jarvisReply, setJarvisReply] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const commandHandlerRef = useRef<(command: string) => void>(() => {});

  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const [activeTab, setActiveTab] = useState<TabId>('briefing');

  // Preferences form
  const [prefName, setPrefName] = useState('');
  const [newInterest, setNewInterest] = useState('');
  const [newSource, setNewSource] = useState('');

  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [serverMeta, setServerMeta] = useState<ServerStatus | null>(null);

  const addLog = useCallback((message: string) => {
    setTerminalLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
  }, []);

  const checkServerStatus = useCallback(async () => {
    try {
      setServerMeta(await api.getStatus());
      setServerOnline(true);
    } catch {
      setServerOnline(false);
    }
  }, []);

  const loadData = useCallback(async () => {
    try {
      const storedPreferences = await api.getPreferences();
      setPreferences(storedPreferences);
      setPrefName(storedPreferences.name);
    } catch (e) {
      console.error('Failed to load preferences:', e);
    }

    try {
      const { history: entries } = await api.getHistory();
      setHistory(entries ?? []);
      if (entries?.length) {
        setBriefing(entries[entries.length - 1].briefing);
      }
    } catch (e) {
      console.error('Failed to load briefing history:', e);
    }
  }, []);

  // Mount-only: fetching preferences here also *sets* them, so depending on
  // `preferences` would re-run this effect on every load and loop forever.
  useEffect(() => {
    // Both are async: state is set from the resolved fetch, not synchronously
    // during the effect body, so no cascading render occurs. The rule cannot
    // see through the promise boundary.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    checkServerStatus();
    loadData();

    const hudInterval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
      setCoreTemp(prev => +(prev + (Math.random() * 0.4 - 0.2)).toFixed(1));
      setMemoryUsage(prev => +(prev + (Math.random() * 1.2 - 0.6)).toFixed(1));
    }, HUD_TICK_MS);
    const serverInterval = setInterval(checkServerStatus, SERVER_POLL_MS);

    return () => {
      clearInterval(hudInterval);
      clearInterval(serverInterval);
      window.speechSynthesis?.cancel();
    };
  }, [checkServerStatus, loadData]);

  // Recognition subscribes once and must always reach the *current* command
  // handler; routing through a ref keeps the subscription stable while the
  // handler it invokes stays fresh. The ref is populated by an effect declared
  // below `handleVoiceCommand`.
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
      addLog('Voice sensors activated. Listening...');
    };
    recognition.onresult = event => {
      const transcript = event.results[0][0].transcript;
      setUserTranscript(transcript);
      addLog(`Command captured: "${transcript}"`);
      commandHandlerRef.current(transcript);
    };
    recognition.onerror = event => {
      addLog(`Voice error: ${event.error}`);
      setIsListening(false);
    };
    recognition.onend = () => setIsListening(false);

    recognitionRef.current = recognition;
    return () => recognition.abort?.();
  }, [addLog]);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalLogs]);

  /** Speak `text` in the JARVIS voice, stripping markdown the synthesiser would read aloud. */
  const speakText = (text: string) => {
    if (!voiceActive || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    const spoken = text.replace(/[*•━—]/g, ' ').replace(/\s+/g, ' ').trim();
    if (!spoken) return;

    const utterance = new SpeechSynthesisUtterance(spoken);
    const voices = window.speechSynthesis.getVoices();
    const preferred =
      PREFERRED_VOICES.map(name => voices.find(v => v.name.includes(name))).find(Boolean) ??
      voices.find(v => v.lang.startsWith('en-GB'));
    if (preferred) utterance.voice = preferred;

    utterance.rate = 1.05;
    utterance.pitch = 0.95;
    utterance.onstart = () => setIsJarvisSpeaking(true);
    utterance.onend = () => setIsJarvisSpeaking(false);
    utterance.onerror = () => setIsJarvisSpeaking(false);

    speechUtteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  };

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      setUserTranscript(null);
      setJarvisReply(null);
      recognitionRef.current?.start();
    }
  };

  const sendAskJarvisRequest = async (message: string) => {
    try {
      addLog(`Routing message to cognitive core: "${message}"`);
      const { response, source } = await api.sendChatMessage(message);
      setJarvisReply(response);
      addLog(`Response received (${source} core).`);
      speakText(response);
    } catch (e) {
      console.error(e);
      addLog('Error contacting cognitive server.');
      speakText('I am having difficulty connecting to my cognitive subroutines, Sir.');
    }
  };

  const handleAskJarvisSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!askJarvisQuery.trim()) return;
    setUserTranscript(askJarvisQuery);
    sendAskJarvisRequest(askJarvisQuery);
    setAskJarvisQuery('');
  };

  /**
   * Run the six-agent workflow. The pipeline is a single blocking call, so the
   * console shows which stage is expected next; once it returns, the real
   * per-node timings from LangGraph replace the estimates.
   */
  const triggerBriefing = async () => {
    setIsRunning(true);
    setError(null);
    setTerminalLogs([]);
    setTelemetry([]);
    addLog('SYS.BOOT: daily briefing routine initiated...');

    const stages = [
      'SYS.PLANNER: reading operator preferences and briefing archive...',
      'SYS.RETRIEVAL: querying NewsAPI and arXiv through MCP tools...',
      'SYS.DEDUP: embedding headlines for semantic duplicate removal...',
      'SYS.RANKING: scoring relevance with local zero-shot classifier...',
      'SYS.RESEARCH: synthesising cross-source insight...',
      'SYS.SUMMARY: composing briefing...',
      'SYS.DELIVERY: formatting and archiving output...',
    ];
    const timers = stages.map((message, index) =>
      window.setTimeout(() => addLog(message), (index + 1) * 1200),
    );
    const clearTimers = () => timers.forEach(window.clearTimeout);

    try {
      const data = await api.runBriefing();
      clearTimers();

      setBriefing(data.briefing);
      setArticles(data.articles ?? []);
      setPapers(data.papers ?? []);
      setTelemetry(data.telemetry ?? []);
      setLastRunSeconds(data.duration_seconds);

      data.telemetry?.forEach(entry =>
        addLog(`SYS.TIMING: ${entry.node} completed in ${entry.seconds.toFixed(2)}s`),
      );
      addLog(`SYS.STATUS: complete in ${data.duration_seconds.toFixed(1)}s — archived to memory.`);
      speakText('Briefing compilation complete, Sir. I have loaded the viewport.');
      loadData();
    } catch (e) {
      clearTimers();
      const message = describeError(e, 'Workflow execution failed.');
      addLog(`SYS.FAIL: core routine aborted — ${message}`);
      setError(message);
      speakText("I was unable to compile today's briefing, Sir.");
    } finally {
      setIsRunning(false);
    }
  };

  /**
   * Interpret a spoken utterance. A few phrases drive the UI directly; anything
   * else is forwarded to the chat endpoint.
   */
  const handleVoiceCommand = (command: string) => {
    const lower = command.toLowerCase();
    const matches = (...phrases: string[]) => phrases.some(phrase => lower.includes(phrase));

    if (matches('briefing', 'compile', 'generate report', 'run brief')) {
      speakText("Compiling today's intelligence briefing, Sir.");
      triggerBriefing();
      return;
    }
    if (matches('mute', 'silence', 'stop talking')) {
      setVoiceActive(false);
      setIsJarvisSpeaking(false);
      window.speechSynthesis?.cancel();
      addLog('Voice synthesis deactivated.');
      return;
    }
    if (matches('unmute', 'talk to me', 'activate speech')) {
      setVoiceActive(true);
      addLog('Voice synthesis activated.');
      setTimeout(() => speakText('Voice synthesisers online. Standing by, Sir.'), 100);
      return;
    }

    sendAskJarvisRequest(command);
  };

  // Keep the recognition subscription pointed at the latest handler.
  useEffect(() => {
    commandHandlerRef.current = handleVoiceCommand;
  });

  const savePreferences = async () => {
    try {
      const stored = await api.savePreferences({ ...preferences, name: prefName });
      setPreferences(stored);
      addLog('Preferences updated successfully.');
      speakText('Preference schema updated, Sir.');
    } catch (e) {
      console.error(e);
      setError(`Could not save preferences: ${describeError(e, 'unknown error')}`);
    }
  };

  const handleAddInterest = () => {
    if (newInterest.trim() && !preferences.interests.includes(newInterest.trim())) {
      const updated = {
        ...preferences,
        interests: [...preferences.interests, newInterest.trim()]
      };
      setPreferences(updated);
      setNewInterest('');
    }
  };

  const handleRemoveInterest = (index: number) => {
    const updated = {
      ...preferences,
      interests: preferences.interests.filter((_, i) => i !== index)
    };
    setPreferences(updated);
  };

  const handleAddSource = () => {
    if (newSource.trim() && !preferences.favorite_sources.includes(newSource.trim())) {
      const updated = {
        ...preferences,
        favorite_sources: [...preferences.favorite_sources, newSource.trim()]
      };
      setPreferences(updated);
      setNewSource('');
    }
  };

  const handleRemoveSource = (index: number) => {
    const updated = {
      ...preferences,
      favorite_sources: preferences.favorite_sources.filter((_, i) => i !== index)
    };
    setPreferences(updated);
  };

  /**
   * Apply the active category and search filters to a feed. `haystack` picks
   * the searchable text, which differs between articles and papers.
   */
  const applyFilters = <T,>(items: T[], haystack: (item: T) => string): T[] => {
    const keywords = selectedCategory === 'All' ? null : (CATEGORY_KEYWORDS[selectedCategory] ?? []);
    const query = searchQuery.trim().toLowerCase();

    return items.filter(item => {
      const text = haystack(item).toLowerCase();
      if (keywords && !keywords.some(keyword => text.includes(keyword))) return false;
      return !query || text.includes(query);
    });
  };

  const getFilteredArticles = () =>
    applyFilters(articles, a => `${a.title ?? ''} ${a.description ?? ''}`);

  const getFilteredPapers = () => applyFilters(papers, p => `${p.title ?? ''} ${p.summary ?? ''}`);

  const renderBriefingContent = (text: string) => {
    if (!text) return null;
    const lines = text.split('\n');
    return (
      <div className="briefing-styled font-code">
        {lines.map((line, idx) => {
          const trimmed = line.trim();
          if (!trimmed) return <div key={idx} className="h-4" />;
          
          if (trimmed.startsWith('━') || trimmed.startsWith('—')) {
            return <div key={idx} className="border-t border-dashed border-purple-500/20 my-4" />;
          }
          
          if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
            const clean = trimmed.replace(/\*\*/g, '');
            if (clean.includes(':')) {
              return (
                <div key={idx} className="mt-6 mb-2">
                  <span className="text-stark-fuchsia font-bold text-sm tracking-wider uppercase border-b border-fuchsia-500/30 pb-1">{clean}</span>
                </div>
              );
            } else {
              return (
                <div key={idx} className="border-l-4 border-purple-500 pl-4 py-2 bg-purple-950/20 rounded-r my-4">
                  <h3 className="text-purple-400 font-semibold text-lg tracking-wide">{clean}</h3>
                </div>
              );
            }
          }
          
          if (trimmed.startsWith('•') || trimmed.startsWith('-')) {
            const clean = trimmed.substring(1).trim();
            const parts = clean.split('**');
            return (
              <div key={idx} className="flex gap-2 items-start py-2 border-b border-slate-900/50">
                <span className="text-purple-400 select-none mt-1">▶</span>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {parts.map((part, pIdx) => {
                    if (pIdx % 2 !== 0) {
                      return <strong key={pIdx} className="text-white font-medium">{part}</strong>;
                    }
                    return part;
                  })}
                </p>
              </div>
            );
          }
          
          return (
            <p key={idx} className="text-slate-300 text-sm leading-relaxed py-1">
              {line}
            </p>
          );
        })}
      </div>
    );
  };

  const filteredArticles = getFilteredArticles();
  const filteredPapers = getFilteredPapers();

  return (
    <div className="relative min-h-screen pb-12 z-10 flex flex-col">
      <div className="grid-bg" />
      
      {/* Stark Immersive Full-Panel Diagnostics Loading Screen */}
      {isRunning && (
        <div className="diagnostics-loader">
          <div className="diagnostics-reactor">
            <div className="arc-ring arc-ring-outer" />
            <div className="arc-ring arc-ring-mid" />
            <div className="arc-ring arc-ring-inner" />
            <div className="arc-core" />
          </div>
          
          <div className="text-center">
            <h2 className="text-purple-400 text-lg uppercase tracking-widest font-extrabold mb-1 pulsing-purple">
              LOADING SYSTEM DATA
            </h2>
            <p className="text-xs text-stark-fuchsia font-code">
              CORE REACTOR LOAD: {(memoryUsage * 1.5).toFixed(1)}% // MCP_TUNNELS_CONNECTED
            </p>
          </div>
          
          <div className="diagnostics-log-box">
            {terminalLogs.map((log, idx) => (
              <div key={idx} className="diagnostic-line">
                <span className="diagnostic-tag">&gt;&gt;</span>
                <span>{log}</span>
              </div>
            ))}
            <div ref={terminalEndRef} />
          </div>
        </div>
      )}

      {/* Unified Control Center Header */}
      <header className="border-b border-purple-500/20 bg-slate-950/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col lg:flex-row justify-between items-stretch lg:items-center gap-4">
          
          {/* Brand Info & Telemetry HUD */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center lg:justify-start gap-4 lg:gap-8 flex-grow">
            <div className="flex items-center gap-3">
              <div className="h-11 w-11 rounded-full border border-purple-500/30 flex items-center justify-center bg-purple-950/20 shadow-[0_0_15px_rgba(168,85,247,0.15)] relative shrink-0">
                <div className="absolute inset-1 rounded-full border border-dashed border-purple-400 animate-spin" />
                <div className="h-4 w-4 rounded-full bg-purple-400 pulsing-purple" />
              </div>
              <div>
                <h1 className="text-xl font-black bg-gradient-to-r from-purple-400 via-fuchsia-300 to-blue-400 bg-clip-text text-transparent tracking-widest leading-none mb-1">
                  J.A.R.V.I.S. Command Feed
                </h1>
                <p className="text-[9px] font-code text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                  <Shield className="h-3 w-3 text-purple-500" /> operator: {preferences.name} // SYSTEM SECURE {serverMeta && `// ENGINE: ${serverMeta.offline_mode ? 'LOCAL (offline)' : serverMeta.llm}`}
                </p>
              </div>
            </div>
            
            {/* HUD Status Bar */}
            <div className="flex items-center gap-4 sm:gap-6 font-code text-[11px] text-slate-400 bg-slate-900/40 border border-purple-500/10 px-4 py-2 rounded">
              <div className="flex items-center gap-2">
                <Activity className="h-3.5 w-3.5 text-purple-400" />
                <span>TEMP: <span className="text-purple-400 font-bold">{coreTemp}°C</span></span>
              </div>
              <div className="flex items-center gap-2 border-l border-purple-500/20 pl-4">
                <Cpu className="h-3.5 w-3.5 text-fuchsia-400" />
                <span>MEM: <span className="text-fuchsia-400 font-bold">{memoryUsage}%</span></span>
              </div>
              <div className="flex items-center gap-2 border-l border-purple-500/20 pl-4">
                <Clock className="h-3.5 w-3.5 text-blue-400" />
                <span>CLOCK: <span className="text-blue-400 font-bold">{currentTime}</span></span>
              </div>
            </div>
          </div>

          {/* Connection & Key Alerts */}
          {serverOnline === false && (
            <div className="flex items-center gap-2 bg-red-950/30 border border-red-500/30 px-3 py-1.5 rounded text-[10px] font-code text-red-400">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
              <span>SERVER OFFLINE — RUN `uvicorn api.main:app --reload`</span>
            </div>
          )}

          {serverOnline === true && serverMeta && !serverMeta.gemini_enabled && (
            <div className="flex items-center gap-2 bg-amber-950/30 border border-amber-500/30 px-3 py-1.5 rounded text-[10px] font-code text-amber-400">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse" />
              <span>GEMINI KEY MISSING — RUNNING ON LOCAL MODELS ONLY</span>
            </div>
          )}

          {serverOnline === true && serverMeta && !serverMeta.news_api_enabled && (
            <div className="flex items-center gap-2 bg-sky-950/30 border border-sky-500/30 px-3 py-1.5 rounded text-[10px] font-code text-sky-400">
              <span className="h-1.5 w-1.5 rounded-full bg-sky-500 animate-pulse" />
              <span>NEWSAPI KEY MISSING — FEED SHOWS LABELLED SAMPLE DATA</span>
            </div>
          )}

        </div>

        {/* Command Inputs Strip */}
        <div className="border-t border-purple-500/10 bg-slate-950/40">
          <div className="max-w-7xl mx-auto px-6 py-3 grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
            
            {/* Search Box */}
            <div className="md:col-span-4 relative flex items-center">
              <Search className="h-3.5 w-3.5 text-purple-500 absolute left-3.5" />
              <input 
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input-glow pl-10 text-xs py-2 bg-slate-900/60"
                placeholder="Search feed database..."
              />
            </div>

            {/* Ask Jarvis Input */}
            <form onSubmit={handleAskJarvisSubmit} className="md:col-span-5 relative flex items-center">
              <Brain className="h-3.5 w-3.5 text-fuchsia-500 absolute left-3.5" />
              <input 
                type="text"
                value={askJarvisQuery}
                onChange={(e) => setAskJarvisQuery(e.target.value)}
                className="input-glow pl-10 pr-22 text-xs py-2 bg-slate-900/60"
                placeholder="Ask Jarvis (e.g. What are my current interests?)"
              />
              <button 
                type="submit"
                className="absolute right-2 px-3 py-1 bg-gradient-to-r from-purple-950 to-fuchsia-950 border border-purple-500/40 hover:border-fuchsia-500/60 rounded text-[9px] font-hud text-purple-300 hover:text-white transition-all cursor-pointer shadow-[0_0_8px_rgba(168,85,247,0.15)]"
              >
                EXECUTE
              </button>
            </form>

            {/* Scope Selector */}
            <div className="md:col-span-3 flex justify-end items-center gap-2">
              <Filter className="h-3.5 w-3.5 text-purple-400 shrink-0" />
              <select 
                value={globalFilter}
                onChange={(e) => setGlobalFilter(e.target.value)}
                className="input-glow text-xs py-2 pr-8 bg-slate-900/60 border-purple-500/20 max-w-[150px] font-code"
              >
                <option value="Worldwide">Worldwide Scope</option>
                <option value="US">US Scope Only</option>
                <option value="UK">UK Scope Only</option>
              </select>
            </div>

          </div>
        </div>
      </header>

      {/* Main Grid Workspace */}
      <main className="max-w-7xl mx-auto px-6 mt-6 flex-grow w-full grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        
        {/* Left Sidebar: Categories + Voice Controls */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          
          {/* Central Holographic spin widget */}
          <div className="glass-panel p-4 flex flex-col items-center justify-center text-center bg-gradient-to-b from-purple-950/5 to-slate-950/20">
            <div className="arc-container mb-3 scale-90">
              <div className="arc-ring arc-ring-outer" />
              <div className="arc-ring arc-ring-mid" />
              <div className="arc-ring arc-ring-inner" />
              <div className={`arc-core ${isListening ? 'alert' : ''}`} />
            </div>
            <div className="font-hud text-[11px]">
              <p className="text-purple-400 font-bold uppercase tracking-wider mb-0.5">
                {isJarvisSpeaking ? "JARVIS VOCAL CORE ACTIVE" : isListening ? "CAPTURING TRANSCRIPTION" : "reactor sync: connected"}
              </p>
              <p className="text-slate-500 text-[9px] font-code">
                AUXILIARY SYS POWER: ONLINE
              </p>
            </div>
          </div>

          {/* Voice Sensors controls */}
          <div className="glass-panel p-4 flex flex-col gap-4 border-purple-500/10">
            <h3 className="text-[10px] uppercase font-bold text-slate-400 tracking-wider flex items-center justify-between font-hud">
              <span>Voice Sensor Controls</span>
              <span className={`h-1.5 w-1.5 rounded-full ${isListening ? 'bg-fuchsia-500 pulsing-purple' : 'bg-slate-700'}`} />
            </h3>
            
            <div className="flex gap-3 justify-center items-center py-1">
              <button
                onClick={toggleListening}
                disabled={!serverOnline || !speechSupported}
                className={`h-12 w-12 rounded-full border flex items-center justify-center transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                  isListening
                    ? 'bg-fuchsia-950/20 border-fuchsia-500 text-fuchsia-400 shadow-[0_0_15px_rgba(217,70,239,0.3)] animate-pulse'
                    : 'bg-purple-950/10 border-purple-500/30 text-purple-400 hover:bg-purple-950/20 hover:border-purple-400'
                }`}
                title={
                  !speechSupported
                    ? 'Speech recognition is unavailable in this browser'
                    : isListening
                      ? 'Mute sensor'
                      : 'Activate sensor'
                }
              >
                {isListening ? <Mic className="h-5 w-5" /> : <MicOff className="h-5 w-5" />}
              </button>

              <button 
                onClick={() => setVoiceActive(!voiceActive)}
                className={`h-10 w-10 rounded-full border flex items-center justify-center transition-all ${
                  voiceActive 
                    ? 'bg-slate-900 border-slate-750 text-purple-400 hover:text-purple-300' 
                    : 'bg-rose-950/10 border-rose-500/30 text-rose-400'
                }`}
                title={voiceActive ? "Silence Jarvis Vocal Output" : "Activate Jarvis Vocal Output"}
              >
                {voiceActive ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
              </button>

              {/* Bouncing audio waveforms */}
              <div className="audio-visualizer ml-1">
                {[...Array(8)].map((_, i) => (
                  <div 
                    key={i} 
                    className={`visualizer-bar ${isJarvisSpeaking ? 'talking' : ''} ${isListening ? 'listening' : ''}`} 
                  />
                ))}
              </div>
            </div>

            {!speechSupported && (
              <p className="text-[9px] font-code text-amber-400/80 leading-relaxed text-center">
                Speech recognition unavailable in this browser — use the text input above.
              </p>
            )}

            {/* Vocal console feeds */}
            {(userTranscript || jarvisReply) && (
              <div className="p-3 bg-slate-950/80 border border-slate-900 rounded flex flex-col gap-2 font-code text-[11px] leading-relaxed">
                {userTranscript && (
                  <div>
                    <span className="text-slate-500 uppercase text-[8px] block">Captured Audio Input:</span>
                    <span className="text-slate-300">"{userTranscript}"</span>
                  </div>
                )}
                {jarvisReply && (
                  <div className="border-t border-slate-900/60 pt-1.5">
                    <span className="text-purple-400 uppercase text-[8px] block">Jarvis vocal response:</span>
                    <span className="text-purple-300 font-medium">{jarvisReply}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Categories Sidebar */}
          <div className="glass-panel p-4 flex flex-col gap-2.5">
            <h3 className="text-[10px] uppercase font-bold text-slate-400 tracking-wider font-hud pb-1 border-b border-purple-500/10 mb-1">
              AI news Categories
            </h3>
            <div className="flex flex-col gap-1">
              {CATEGORIES.map((cat) => (
                <button 
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`flex items-center justify-between w-full px-3 py-2 rounded text-left font-hud text-xs tracking-wider transition-all border ${
                    selectedCategory === cat 
                      ? 'bg-purple-950/20 border-purple-500/50 text-purple-400 font-bold shadow-[0_0_8px_rgba(168,85,247,0.08)]' 
                      : 'bg-transparent border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
                  }`}
                >
                  <span>{cat}</span>
                  {selectedCategory === cat && <ChevronRight className="h-3.5 w-3.5" />}
                </button>
              ))}
            </div>
          </div>

          {/* Tab switcher */}
          <div className="glass-panel p-3 flex gap-2">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 py-2 text-center border text-[10px] font-hud rounded tracking-wider ${
                  activeTab === tab.id
                    ? 'bg-purple-950/20 border-purple-500/50 text-purple-400'
                    : 'border-transparent text-slate-400 hover:bg-slate-900/20'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

        </div>

        {/* Center Feed & Right Panel Column */}
        <div className="lg:col-span-9 flex flex-col gap-6">
          
          {activeTab === 'briefing' && (
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-stretch flex-grow">
              
              {/* Central News Feed (Column 7) */}
              <div className="md:col-span-7 flex flex-col gap-4">
                <div className="glass-panel flex-grow flex flex-col min-h-[500px]">
                  <div className="flex justify-between items-center border-b border-slate-900 pb-3 mb-4">
                    <div className="flex items-center gap-2">
                      <BookOpen className="text-purple-400 h-4.5 w-4.5" />
                      <h2 className="text-sm font-bold text-slate-200">Global AI news headlines</h2>
                    </div>
                    <span className="text-[10px] font-code text-purple-400 bg-purple-950/20 border border-purple-500/20 px-2 py-0.5 rounded uppercase">
                      {filteredArticles.length + filteredPapers.length} articles active
                    </span>
                  </div>

                  <div className="flex-grow overflow-y-auto max-h-[550px] flex flex-col gap-4 pr-1">
                    {error && (
                      <div className="p-3 bg-rose-950/25 border border-rose-500/30 rounded text-rose-400 text-xs font-code">
                        SYSTEM EXCEPTION: {error}
                      </div>
                    )}
                    {filteredArticles.length === 0 && filteredPapers.length === 0 ? (
                      <div className="flex flex-col items-center justify-center text-center h-full min-h-[300px] text-slate-500 gap-3">
                        <Compass className="h-12 w-12 text-slate-800" />
                        <div>
                          <p className="font-hud text-xs text-slate-400 mb-0.5">No signal feeds found</p>
                          <p className="text-[10px] font-code max-w-xs leading-relaxed">
                            No articles match the category "{selectedCategory}" in your cache. Speak "compile briefing" or click compile to fetch latest updates.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <>
                        {/* Articles Grid Cards */}
                        {filteredArticles.map((item, idx) => (
                          <a 
                            href={item.url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            key={idx}
                            // shrink-0: this is a flex-column child, so without it
                            // the cards compress below their content height and
                            // clip their own titles once the feed fills up.
                            className="block shrink-0 p-4 border border-slate-900 bg-slate-950/60 hover:bg-slate-950/20 rounded transition-all hover:border-purple-500/30 group relative overflow-hidden"
                          >
                            <div className="absolute top-0 right-0 h-1.5 w-1.5 bg-purple-500/30" />
                            <div className="flex justify-between items-start gap-2 mb-2 font-code text-[10px]">
                              <span className="text-purple-400 bg-purple-950/20 border border-purple-500/10 px-1.5 py-0.5 rounded">
                                {item.source}
                              </span>
                              {item.relevance_score !== undefined && (
                                <span className="text-fuchsia-400 font-bold text-[9px] tracking-wider">
                                  RELEVANCE: {(item.relevance_score * 100).toFixed(0)}% MATCH
                                </span>
                              )}
                            </div>
                            <h4 className="text-sm text-slate-200 font-hud font-bold group-hover:text-purple-400 transition-colors leading-snug mb-1.5">
                              {item.title}
                            </h4>
                            <p className="text-xs text-slate-400 leading-relaxed font-sans font-light">
                              {item.description}
                            </p>
                            {item.published && (
                              <div className="text-[9px] text-slate-500 font-code mt-2">
                                TIMESTAMP: {item.published.substring(0, 10)} // STATUS: ACTIVE
                              </div>
                            )}
                          </a>
                        ))}

                        {/* Research Papers Grid Cards */}
                        {filteredPapers.map((item, idx) => (
                          <a 
                            href={item.url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            key={idx}
                            className="block shrink-0 p-4 border border-slate-900 bg-slate-950/60 hover:bg-slate-950/20 rounded transition-all hover:border-fuchsia-500/30 group relative overflow-hidden"
                          >
                            <div className="absolute top-0 right-0 h-1.5 w-1.5 bg-fuchsia-500/30" />
                            <div className="flex justify-between items-start gap-2 mb-2 font-code text-[10px]">
                              <span className="text-fuchsia-400 bg-fuchsia-950/20 border border-fuchsia-500/10 px-1.5 py-0.5 rounded">
                                arXiv preprint
                              </span>
                              <span className="text-slate-500">
                                {item.published}
                              </span>
                            </div>
                            <h4 className="text-sm text-slate-200 font-hud font-bold group-hover:text-fuchsia-400 transition-colors leading-snug mb-1.5">
                              {item.title}
                            </h4>
                            <p className="text-xs text-slate-400 leading-relaxed font-sans font-light">
                              {item.summary}
                            </p>
                            <div className="text-[9px] text-slate-500 font-code mt-2">
                              AUTHORS: {item.authors ? item.authors.join(', ') : 'Unknown'}
                            </div>
                          </a>
                        ))}
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Right Panel: Live Insights & Digest Mode (Column 5) */}
              <div className="md:col-span-5 flex flex-col gap-6">
                
                {/* Digest Mode Summarizer Box */}
                <div className="glass-panel flex-grow flex flex-col border-purple-500/20 shadow-[0_0_15px_rgba(168,85,247,0.03)]">
                  <div className="flex justify-between items-center border-b border-slate-900 pb-3 mb-3">
                    <div className="flex items-center gap-2">
                      <Brain className="text-purple-400 h-4.5 w-4.5" />
                      <h2 className="text-sm font-bold text-slate-200">Jarvis digest report</h2>
                    </div>
                    <button 
                      onClick={triggerBriefing}
                      disabled={isRunning || !serverOnline}
                      className="px-2 py-0.5 border border-fuchsia-500/30 text-fuchsia-400 hover:bg-fuchsia-950/20 hover:text-white rounded text-[8px] font-hud transition-colors"
                    >
                      DIGEST
                    </button>
                  </div>

                  <div className="flex-grow overflow-y-auto max-h-[300px] text-xs font-code pr-1">
                    {briefing ? (
                      renderBriefingContent(briefing)
                    ) : (
                      <div className="flex flex-col items-center justify-center text-center h-full min-h-[150px] text-slate-500 gap-2">
                        <FileText className="h-8 w-8 text-slate-800" />
                        <p className="text-[10px] font-code uppercase">DIGEST CACHE EMPTY</p>
                        <p className="text-[9px] font-code max-w-[200px] leading-relaxed">
                          Click DIGEST in the header to run summarization subroutines.
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Agent pipeline telemetry — real per-node timings from LangGraph */}
                <div className="glass-panel p-4 flex flex-col gap-4">
                  <h3 className="text-xs font-bold text-slate-200 tracking-wider font-hud flex items-center justify-between gap-1.5">
                    <span className="flex items-center gap-1.5">
                      <TrendingUp className="h-4 w-4 text-purple-400" />
                      <span>Agent Pipeline Telemetry</span>
                    </span>
                    {lastRunSeconds !== null && (
                      <span className="text-[9px] font-code text-slate-500">
                        TOTAL {lastRunSeconds.toFixed(1)}s
                      </span>
                    )}
                  </h3>

                  {telemetry.length === 0 ? (
                    <p className="text-[10px] font-code text-slate-500 leading-relaxed py-2">
                      No run recorded this session. Compile a briefing to capture per-agent
                      execution timings from the LangGraph workflow.
                    </p>
                  ) : (
                    <div className="flex flex-col gap-3">
                      {telemetry.map(entry => {
                        const slowest = Math.max(...telemetry.map(t => t.seconds), 0.001);
                        const share = Math.round((entry.seconds / slowest) * 100);
                        const failed = entry.status !== 'ok';
                        return (
                          <div
                            key={entry.node}
                            className="flex flex-col gap-1.5 p-2.5 border border-slate-900 bg-slate-950/40 rounded"
                          >
                            <div className="flex justify-between items-center">
                              <span className="text-xs font-hud text-slate-200 font-bold tracking-wider uppercase">
                                {entry.node}
                              </span>
                              <span
                                className={`text-[9px] font-code px-2 py-0.5 rounded border ${
                                  failed
                                    ? 'text-rose-400 bg-rose-950/30 border-rose-500/20'
                                    : 'text-purple-400 bg-purple-950/30 border-purple-500/20'
                                }`}
                              >
                                {entry.seconds.toFixed(2)}s
                              </span>
                            </div>
                            <div className="h-1 w-full bg-slate-900 rounded overflow-hidden">
                              <div
                                className={`h-full ${failed ? 'bg-rose-500/60' : 'bg-purple-500/60'}`}
                                style={{ width: `${share}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

              </div>

            </div>
          )}

          {activeTab === 'history' && (
            <div className="glass-panel flex flex-col gap-6">
              <div className="flex items-center justify-between border-b border-slate-900 pb-4">
                <div className="flex items-center gap-2">
                  <History className="text-purple-400 h-5 w-5" />
                  <h2 className="text-md font-bold uppercase text-slate-200">Briefing History Archive</h2>
                </div>
                <span className="text-xs font-code text-purple-400 bg-purple-950/20 border border-purple-500/20 px-2 py-0.5 rounded">
                  {history.length} Saved Reports
                </span>
              </div>

              {history.length === 0 ? (
                <div className="text-center py-16 text-slate-500 font-code text-sm">
                  No historical reports found in briefing_history.json
                </div>
              ) : (
                <div className="flex flex-col gap-4">
                  {[...history].reverse().map((item, idx) => (
                    <div 
                      key={idx} 
                      className="border border-slate-900 rounded-lg bg-slate-950/30 overflow-hidden transition-all hover:border-slate-800"
                    >
                      <div className="flex justify-between items-center px-4 py-3 bg-slate-950/50 border-b border-slate-900">
                        <span className="font-code text-xs text-slate-400 font-bold">{item.date}</span>
                        <button 
                          onClick={() => {
                            setBriefing(item.briefing);
                            setActiveTab('briefing');
                          }}
                          className="flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300 font-code"
                        >
                          <span>Load Viewport</span>
                          <ChevronRight className="h-3.5 w-3.5" />
                        </button>
                      </div>
                      <div className="p-4 max-h-48 overflow-y-auto text-xs opacity-75 font-code whitespace-pre-wrap leading-relaxed">
                        {item.briefing.substring(0, 400)}...
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'preferences' && (
            <div className="glass-panel flex flex-col gap-6">
              <div className="flex items-center justify-between border-b border-slate-900 pb-4">
                <div className="flex items-center gap-2">
                  <Settings className="text-purple-400 h-5 w-5" />
                  <h2 className="text-md font-bold uppercase text-slate-200">System Preferences Config</h2>
                </div>
                <span className="text-xs font-code text-slate-500">SAVED IN JSON</span>
              </div>

              <div className="flex flex-col gap-6 max-w-xl">
                {/* Operator Name */}
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Operator Name</label>
                  <div className="flex items-center gap-2">
                    <User className="text-purple-500 h-4 w-4 absolute ml-3" />
                    <input 
                      type="text" 
                      value={prefName}
                      onChange={(e) => setPrefName(e.target.value)}
                      className="input-glow pl-10"
                      placeholder="Enter operator name..."
                    />
                  </div>
                </div>

                {/* Focus Interests */}
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Focus Topics / Interests</label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {preferences.interests.map((interest, idx) => (
                      <span 
                        key={idx} 
                        className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-purple-950/20 border border-purple-500/30 text-xs text-purple-300 font-code"
                      >
                        {interest}
                        <button 
                          onClick={() => handleRemoveInterest(idx)}
                          className="hover:text-rose-400 focus:outline-none"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                  
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={newInterest}
                      onChange={(e) => setNewInterest(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleAddInterest()}
                      className="input-glow text-sm"
                      placeholder="Add interest topic..."
                    />
                    <button 
                      onClick={handleAddInterest}
                      className="px-4 py-2 bg-slate-900 border border-purple-500/30 text-purple-400 rounded hover:bg-slate-800 transition-colors"
                    >
                      <Plus className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Favorite Sources */}
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Favorite Media Sources</label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {preferences.favorite_sources.map((source, idx) => (
                      <span 
                        key={idx} 
                        className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-slate-950/20 border border-slate-800/40 text-xs text-slate-300 font-code"
                      >
                        {source}
                        <button 
                          onClick={() => handleRemoveSource(idx)}
                          className="hover:text-rose-400 focus:outline-none"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>

                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={newSource}
                      onChange={(e) => setNewSource(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleAddSource()}
                      className="input-glow text-sm"
                      placeholder="Add media source..."
                    />
                    <button 
                      onClick={handleAddSource}
                      className="px-4 py-2 bg-slate-900 border border-slate-800/40 text-purple-400 rounded hover:bg-slate-800 transition-colors"
                    >
                      <Plus className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                <div className="border-t border-slate-900 pt-4 mt-2">
                  <button 
                    onClick={savePreferences}
                    className="btn-glow text-xs"
                  >
                    Save Preference Schema
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
