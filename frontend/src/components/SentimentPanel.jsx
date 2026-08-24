import React from "react";
import { Newspaper, TrendingUp, TrendingDown, Minus, ExternalLink } from "lucide-react";
import { useTheme } from "../lib/theme.jsx";

const SENTIMENT_STYLE = {
  POSITIVE: { color: "var(--color-term-green)", bg: "bg-[var(--color-term-green-dim)]", Icon: TrendingUp },
  NEGATIVE: { color: "var(--color-term-red)", bg: "bg-[var(--color-term-red-dim)]", Icon: TrendingDown },
  NEUTRAL: { color: "var(--color-amber)", bg: "bg-[var(--color-amber-dim)]", Icon: Minus },
};

const SOURCE_COLOR_DARK = {
  Yahoo: "text-[var(--color-amber)] border-[var(--color-amber)]/30",
  Google: "text-[var(--color-term-cyan)] border-[var(--color-term-cyan)]/30",
};
const SOURCE_COLOR_LIGHT = {
  Yahoo: "text-purple-700 bg-purple-50",
  Google: "text-blue-700 bg-blue-50",
};

const SentimentPanel = ({ ticker, sentiment, articles, loading }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const label = sentiment?.label ? String(sentiment.label).toUpperCase() : null;
  const style = SENTIMENT_STYLE[label] || SENTIMENT_STYLE.NEUTRAL;
  const Icon = style.Icon;

  return (
    <div className={`flex flex-col h-full bg-[var(--color-panel)] ${isDark ? "border-l border-[var(--color-hairline)]" : "border-l border-[var(--color-hairline)]"}`}>
      <div className="px-4 py-3 border-b border-[var(--color-hairline)] flex items-center gap-2">
        <Newspaper size={15} style={{ color: "var(--color-amber)" }} />
        <h2 className={`text-sm font-semibold text-[var(--color-text-primary)] ${isDark ? "font-mono text-xs tracking-[0.15em]" : ""}`}>
          {isDark ? "NEWS & SENTIMENT" : "News & Sentiment"}
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto term-scroll p-4 space-y-5">
        {/* Overall sentiment */}
        <div
          className={`p-4 text-center rounded-[var(--radius-card)] border border-[var(--color-hairline)]`}
          style={{ boxShadow: isDark ? "none" : "var(--shadow-card)" }}
        >
          <h3 className={`text-[10px] tracking-[0.15em] text-[var(--color-text-dim)] mb-3 uppercase font-semibold`}>
            Overall Sentiment
          </h3>
          {sentiment ? (
            <div className="flex flex-col items-center gap-2">
              <div className={`p-2.5 rounded-full ${style.bg}`} style={{ color: style.color }}>
                <Icon size={22} />
              </div>
              <div className="text-xl font-bold tracking-wide" style={{ color: style.color }}>
                {isDark ? label : label?.charAt(0) + label?.slice(1).toLowerCase()}
              </div>
              <div className="text-xs text-[var(--color-text-dim)] tabular">
                Confidence {(sentiment.score * 100).toFixed(1)}%
              </div>
            </div>
          ) : (
            <div className="text-[var(--color-text-faint)] text-xs italic py-4">
              Run analysis to compute sentiment.
            </div>
          )}
        </div>

        {/* Headlines */}
        <div>
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-[10px] tracking-[0.15em] text-[var(--color-text-dim)] uppercase font-semibold">
              Live Headlines
            </h3>
            <span className="text-[10px] text-[var(--color-text-dim)] border border-[var(--color-hairline)] rounded-full px-2 py-0.5">
              {articles.length}
            </span>
          </div>

          <div className="space-y-2.5">
            {loading ? (
              <div className="text-center py-8 text-[var(--color-text-dim)] text-xs animate-pulse">
                Fetching news...
              </div>
            ) : articles.length > 0 ? (
              articles.map((art, i) => {
                const sourceClass = isDark
                  ? (SOURCE_COLOR_DARK[art.source] || "text-[var(--color-text-dim)] border-[var(--color-hairline)]")
                  : (SOURCE_COLOR_LIGHT[art.source] || "text-[var(--color-text-dim)] bg-[var(--color-void)]");
                return (
                  <a
                    key={i}
                    href={art.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`block p-3 rounded-[var(--radius-card)] border border-[var(--color-hairline)] hover:border-[var(--color-amber)]/50 transition-colors group ${isDark ? "" : "hover:shadow-sm"}`}
                  >
                    <div className="flex justify-between items-start gap-2 mb-1.5">
                      <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${isDark ? "border" : ""} ${sourceClass}`}>
                        {art.source || "WIRE"}
                      </span>
                      <span className="text-[9px] text-[var(--color-text-faint)] whitespace-nowrap">
                        {art.published && art.published !== "Recent" ? new Date(art.published).toLocaleDateString() : "Recent"}
                      </span>
                    </div>
                    <div className="flex items-start gap-1.5">
                      <span className="text-[13px] text-[var(--color-text-primary)] leading-snug group-hover:text-[var(--color-amber)] transition-colors">
                        {art.title}
                      </span>
                      <ExternalLink size={11} className="text-[var(--color-text-faint)] mt-0.5 flex-shrink-0" />
                    </div>
                  </a>
                );
              })
            ) : (
              <div className="text-[var(--color-text-faint)] text-xs italic text-center py-6">
                No recent headlines for {ticker}.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SentimentPanel;
