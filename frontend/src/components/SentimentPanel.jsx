import React from "react";
import { Newspaper, TrendingUp, TrendingDown, Minus, ExternalLink } from "lucide-react";

const SENTIMENT_STYLE = {
  POSITIVE: { color: "var(--color-term-green)", bg: "bg-[var(--color-term-green-dim)]", Icon: TrendingUp },
  NEGATIVE: { color: "var(--color-term-red)", bg: "bg-[var(--color-term-red-dim)]", Icon: TrendingDown },
  NEUTRAL: { color: "var(--color-amber)", bg: "bg-[var(--color-amber-dim)]/30", Icon: Minus },
};

const SOURCE_COLOR = {
  Yahoo: "text-[var(--color-amber)] border-[var(--color-amber)]/30",
  Google: "text-[var(--color-term-cyan)] border-[var(--color-term-cyan)]/30",
};

const SentimentPanel = ({ ticker, sentiment, articles, loading }) => {
  const label = sentiment?.label ? String(sentiment.label).toUpperCase() : null;
  const style = SENTIMENT_STYLE[label] || SENTIMENT_STYLE.NEUTRAL;
  const Icon = style.Icon;

  return (
    <div className="flex flex-col h-full bg-[var(--color-panel)] border-l border-[var(--color-hairline)]">
      <div className="px-4 py-3 border-b border-[var(--color-hairline)] flex items-center gap-2">
        <Newspaper size={15} className="text-[var(--color-amber)]" />
        <h2 className="font-mono text-xs tracking-[0.15em] text-[var(--color-text-primary)]">
          NEWS &amp; SENTIMENT
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto term-scroll p-4 space-y-5">
        <div className="border border-[var(--color-hairline)] p-4 text-center">
          <h3 className="font-mono text-[10px] tracking-[0.2em] text-[var(--color-text-dim)] mb-3">
            OVERALL SENTIMENT
          </h3>
          {sentiment ? (
            <div className="flex flex-col items-center gap-2">
              <div className={`p-2.5 ${style.bg}`} style={{ color: style.color }}>
                <Icon size={22} />
              </div>
              <div className="font-mono text-xl font-bold tracking-wide" style={{ color: style.color }}>
                {label}
              </div>
              <div className="font-mono text-xs text-[var(--color-text-dim)] tabular">
                CONFIDENCE {(sentiment.score * 100).toFixed(1)}%
              </div>
            </div>
          ) : (
            <div className="text-[var(--color-text-faint)] font-mono text-xs italic py-4">
              Run analysis to compute sentiment.
            </div>
          )}
        </div>

        <div>
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-mono text-[10px] tracking-[0.2em] text-[var(--color-text-dim)]">
              LIVE HEADLINES
            </h3>
            <span className="font-mono text-[10px] text-[var(--color-text-dim)] border border-[var(--color-hairline)] px-1.5 py-0.5">
              {articles.length}
            </span>
          </div>

          <div className="space-y-2.5">
            {loading ? (
              <div className="text-center py-8 text-[var(--color-text-dim)] font-mono text-xs animate-pulse">
                FETCHING NEWS...
              </div>
            ) : articles.length > 0 ? (
              articles.map((art, i) => (
                <a
                  key={i}
                  href={art.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block border border-[var(--color-hairline)] p-3 hover:border-[var(--color-amber)]/50 transition-colors group"
                >
                  <div className="flex justify-between items-start gap-2 mb-1.5">
                    <span className={`font-mono text-[9px] font-semibold px-1.5 py-0.5 border ${SOURCE_COLOR[art.source] || "text-[var(--color-text-dim)] border-[var(--color-hairline)]"}`}>
                      {art.source || "WIRE"}
                    </span>
                    <span className="font-mono text-[9px] text-[var(--color-text-faint)] whitespace-nowrap">
                      {art.published && art.published !== "Recent" ? new Date(art.published).toLocaleDateString() : "Recent"}
                    </span>
                  </div>
                  <div className="flex items-start gap-1.5">
                    <span className="font-sans text-[13px] text-[var(--color-text-primary)] leading-snug group-hover:text-[var(--color-amber)] transition-colors">
                      {art.title}
                    </span>
                    <ExternalLink size={11} className="text-[var(--color-text-faint)] mt-0.5 flex-shrink-0" />
                  </div>
                </a>
              ))
            ) : (
              <div className="text-[var(--color-text-faint)] font-mono text-xs italic text-center py-6">
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
