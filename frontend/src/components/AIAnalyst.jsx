import React, { useState, useEffect, useCallback } from "react";
import TopBar from "./components/TopBar";
import SignalReadout from "./components/SignalReadout";
import DeepChart from "./components/DeepChart";
import SentimentPanel from "./components/SentimentPanel";
import AIAnalyst from "./components/AIAnalyst";
import { api, slowApi, describeApiError } from "./lib/api";
import { useTheme } from "./lib/theme.jsx";
import { Newspaper, MessageCircle, Terminal as TerminalIcon, TriangleAlert } from "lucide-react";

const App = () => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const ChatTabIcon = isDark ? TerminalIcon : MessageCircle;

  const [ticker, setTicker] = useState("NVDA");
  const [activeTicker, setActiveTicker] = useState("NVDA");
  const [analysis, setAnalysis] = useState(null);
  const [news, setNews] = useState("");
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [newsLoading, setNewsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sidebarTab, setSidebarTab] = useState("news"); // 'news' | 'chat'

  // PERFORMANCE FIX: this used to await /news, THEN await /analyze --
  // strictly sequential, so total wait time was the SUM of both requests.
  // On a cold Render free-tier instance (backend spun down after 15 min
  // idle), each of those individually can already take 20-40s+, so back
  // -to-back they could take a minute or more before anything appeared.
  //
  // /analyze doesn't actually need the frontend to fetch news first --
  // the backend already has its own fallback (see main.py's analyze_stock:
  // "if not news_context: news_context, _ = get_news_engine()...") that
  // fetches the exact same cached news lookup server-side. So both
  // requests can fire at the same time via Promise.allSettled: the
  // headline list and the signal/sentiment now arrive independently and
  // roughly in parallel instead of one blocking the other. On a cold
  // start this can roughly halve the wait; on a warm backend it's
  // unnoticeable either way.
  //
  // Promise.allSettled (not Promise.all) so one endpoint failing doesn't
  // wipe out a result the other endpoint already got back successfully.
  const runAnalysis = useCallback(async () => {
    const target = ticker.trim();
    if (!target) return;

    setActiveTicker(target);
    setLoading(true);
    setNewsLoading(true);
    setError(null);
    setAnalysis(null);
    setArticles([]);

    const [newsResult, analyzeResult] = await Promise.allSettled([
      api.get(`/news/${target}`),
      slowApi.post("/analyze", { ticker: target }),
    ]);

    if (newsResult.status === "fulfilled") {
      const data = newsResult.value.data;
      setArticles(data.articles || []);
      setNews((prev) => prev || data.context || "");
    }
    setNewsLoading(false);

    if (analyzeResult.status === "fulfilled") {
      setAnalysis(analyzeResult.value.data);
      if (analyzeResult.value.data.news_context_used) {
        setNews(analyzeResult.value.data.news_context_used);
      }
    } else {
      setError(describeApiError(analyzeResult.reason));
    }

    setLoading(false);
  }, [ticker]);

  useEffect(() => {
    runAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const tabLabel = (label) => (isDark ? label.toUpperCase() : label);

  return (
    <div className="min-h-screen bg-[var(--color-void)] text-[var(--color-text-primary)] flex flex-col font-sans">
      <TopBar ticker={ticker} onTickerChange={setTicker} onRun={runAnalysis} loading={loading} />

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto term-scroll p-5 space-y-5">
          {error && (
            <div
              className={`border border-[var(--color-term-red)]/50 bg-[var(--color-term-red-dim)] px-4 py-3 flex items-center gap-2 text-xs rounded-[var(--radius-card)] ${isDark ? "font-mono" : ""}`}
              style={{ color: "var(--color-term-red)" }}
            >
              <TriangleAlert size={14} />
              {error}
            </div>
          )}

          {analysis && (
            <SignalReadout
              signalClass={analysis.signal_class}
              confidence={analysis.confidence}
              ticker={analysis.ticker || activeTicker}
            />
          )}

          <DeepChart ticker={activeTicker} />
        </div>

        <div className="w-96 flex-shrink-0 flex flex-col">
          <div className="flex border-b border-l border-[var(--color-hairline)] bg-[var(--color-panel)]">
            <button
              onClick={() => setSidebarTab("news")}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 text-[11px] tracking-wide transition-colors ${
                isDark ? "font-mono" : "font-medium"
              } ${
                sidebarTab === "news"
                  ? "text-[var(--color-amber)] border-b-2 border-[var(--color-amber)]"
                  : "text-[var(--color-text-dim)] hover:text-[var(--color-text-primary)]"
              }`}
            >
              <Newspaper size={13} /> {tabLabel("News")}
            </button>
            <button
              onClick={() => setSidebarTab("chat")}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 text-[11px] tracking-wide transition-colors ${
                isDark ? "font-mono" : "font-medium"
              } ${
                sidebarTab === "chat"
                  ? "text-[var(--color-amber)] border-b-2 border-[var(--color-amber)]"
                  : "text-[var(--color-text-dim)] hover:text-[var(--color-text-primary)]"
              }`}
            >
              <ChatTabIcon size={13} /> {tabLabel("Analyst")}
            </button>
          </div>

          <div className="flex-1 overflow-hidden">
            {sidebarTab === "news" ? (
              <SentimentPanel ticker={activeTicker} sentiment={analysis?.sentiment} articles={articles} loading={newsLoading} />
            ) : (
              <AIAnalyst ticker={activeTicker} context={news} signal={analysis?.signal_class} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;