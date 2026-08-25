import React, { useEffect, useState } from "react";
import { Activity } from "lucide-react";

/**
 * TopBar -- Bloomberg-style TICKER<GO> command line. Theme toggle removed
 * (app is dark-only now); this is simply the terminal top bar.
 */
const TopBar = ({ ticker, onTickerChange, onRun, loading }) => {
  const [now, setNow] = useState(new Date());
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const timeStr = now.toLocaleTimeString("en-US", { hour12: false });

  const handleKeyDown = (e) => {
    if (e.key === "Enter") onRun();
  };

  return (
    <header className="border-b border-[var(--color-hairline)] bg-[var(--color-panel)] px-5 py-3 flex items-center justify-between sticky top-0 z-20">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-[var(--color-amber)] rounded-full" />
          <span className="font-mono text-sm font-semibold tracking-widest text-[var(--color-text-primary)]">
            SENTIMETRIX<span className="text-[var(--color-amber)]">-TCN</span>
          </span>
        </div>

        <div
          className={`flex items-center font-mono text-sm border px-3 py-1.5 transition-colors ${
            focused ? "border-[var(--color-amber)]" : "border-[var(--color-hairline)]"
          } bg-[var(--color-void)]`}
        >
          <input
            value={ticker}
            onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="TICKER"
            aria-label="Ticker symbol"
            className="bg-transparent outline-none text-[var(--color-text-primary)] w-28 tabular tracking-wider placeholder:text-[var(--color-text-faint)]"
          />
          <span className="blink-cursor text-[var(--color-amber)]">▮</span>
          <button
            onClick={onRun}
            disabled={loading}
            className="ml-3 px-2.5 py-0.5 text-xs font-bold tracking-wide bg-[var(--color-amber)] text-black hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "···" : "<GO>"}
          </button>
        </div>
      </div>

      <div className="flex items-center gap-4 font-mono text-xs text-[var(--color-text-dim)]">
        <div className="flex items-center gap-1.5">
          <Activity size={12} className="text-[var(--color-term-green)]" />
          <span className="text-[var(--color-term-green)] tracking-wide">LIVE</span>
        </div>
        <span className="tabular">{timeStr} LOCAL</span>
      </div>
    </header>
  );
};

export default TopBar;
