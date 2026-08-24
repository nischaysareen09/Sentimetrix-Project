import React, { useState, useEffect, useCallback } from "react";
import TopBar from "./components/TopBar";
import SignalReadout from "./components/SignalReadout";
import DeepChart from "./components/DeepChart";
import SentimentPanel from "./components/SentimentPanel";
import AIAnalyst from "./components/AIAnalyst";
import { api, describeApiError } from "./lib/api";
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
  const [error, setError] = useState(null);
  const [sidebarTab, setSidebarTab] = useState("news"); // 'news' | 'chat'

  const runAnalysis = useCallback(async () => {
    const target = ticker.trim();
    if (!target) return;

    setActiveTicker(target);
    setLoading(true);
    setError(null);

    try {
      const newsRes = await api.get(`/news/${target}`);
      const latestArticles = newsRes.data.articles || [];
      const latestContext = newsRes.data.context || "";
      setArticles(latestArticles);
      setNews(latestContext);

      const res = await api.post("/analyze", {
        ticker: target,
        news_context: latestContext,
      });

      setAnalysis(res.data);
      if (res.data.news_context_used) setNews(res.data.news_context_used);
    } catch (err) {
      setError(describeApiError(err));
    } finally {
      setLoading(false);
    }
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
        {/* Main column */}
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

        {/* Sidebar: tabbed News/Sentiment vs AI Analyst */}
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
              <SentimentPanel ticker={activeTicker} sentiment={analysis?.sentiment} articles={articles} loading={loading} />
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
