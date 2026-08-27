import React, { useState, useEffect, useCallback } from "react";
import TickerTape from "./components/TickerTape";
import TopBar from "./components/TopBar";
import SignalReadout from "./components/SignalReadout";
import DeepChart from "./components/DeepChart";
import SentimentPanel from "./components/SentimentPanel";
import { api, slowApi, describeApiError } from "./lib/api";
import { TriangleAlert } from "lucide-react";

const App = () => {
  const [ticker, setTicker] = useState("NVDA");
  const [activeTicker, setActiveTicker] = useState("NVDA");
  const [analysis, setAnalysis] = useState(null);
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [newsLoading, setNewsLoading] = useState(false);
  const [error, setError] = useState(null);

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
    }
    setNewsLoading(false);

    if (analyzeResult.status === "fulfilled") {
      setAnalysis(analyzeResult.value.data);
    } else {
      setError(describeApiError(analyzeResult.reason));
    }

    setLoading(false);
  }, [ticker]);

  useEffect(() => {
    runAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen bg-[var(--color-void)] text-[var(--color-text-primary)] flex flex-col font-sans">
      <TickerTape />
      <TopBar ticker={ticker} onTickerChange={setTicker} onRun={runAnalysis} loading={loading} />

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto term-scroll p-5 space-y-5">
          {error && (
            <div className="border border-[var(--color-term-red)]/50 bg-[var(--color-term-red-dim)] px-4 py-3 flex items-center gap-2 font-mono text-xs text-[var(--color-term-red)]">
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

        <div className="w-96 flex-shrink-0">
          <SentimentPanel ticker={activeTicker} sentiment={analysis?.sentiment} articles={articles} loading={newsLoading} />
        </div>
      </div>
    </div>
  );
};

export default App;