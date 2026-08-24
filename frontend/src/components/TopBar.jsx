import React, { useEffect, useState } from "react";
import { Activity, Sun, Moon } from "lucide-react";
import { useTheme } from "../lib/theme.jsx";

/**
 * TopBar — branches structurally between the two themes:
 * - light: white header, pill-shaped search (Tickertape-style), blue
 *   primary button, subtle border
 * - dark: the original command-line "TICKER<GO>" terminal bar
 */
const TopBar = ({ ticker, onTickerChange, onRun, loading }) => {
  const { theme, toggleTheme } = useTheme();
  const [now, setNow] = useState(new Date());
  const [focused, setFocused] = useState(false);
  const isDark = theme === "dark";

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const timeStr = now.toLocaleTimeString("en-US", { hour12: false });

  const handleKeyDown = (e) => {
    if (e.key === "Enter") onRun();
  };

  return (
    <header
      className={`px-5 py-3 flex items-center justify-between sticky top-0 z-20 border-b ${
        isDark
          ? "border-[var(--color-hairline)] bg-[var(--color-panel)]"
          : "border-[var(--color-hairline)] bg-white/90 backdrop-blur-md"
      }`}
    >
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${isDark ? "bg-[var(--color-amber)]" : "bg-[var(--color-amber)]"}`}
          />
          <span
            className={`text-sm font-bold tracking-tight ${isDark ? "font-mono tracking-widest" : ""}`}
            style={{ color: "var(--color-text-primary)" }}
          >
            SENTIMETRIX<span style={{ color: "var(--color-amber)" }}>-TCN</span>
          </span>
        </div>

        {isDark ? (
          // Dark theme: Bloomberg-style command line
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
        ) : (
          // Light theme: pill-shaped search, Tickertape-style
          <div
            className={`flex items-center rounded-full border px-4 py-2 transition-colors ${
              focused ? "border-[var(--color-amber)] shadow-sm" : "border-[var(--color-hairline)]"
            } bg-[var(--color-void)]`}
          >
            <input
              value={ticker}
              onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
              onKeyDown={handleKeyDown}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder="Search a stock, e.g. NVDA"
              aria-label="Ticker symbol"
              className="bg-transparent outline-none text-sm w-44 text-[var(--color-text-primary)] placeholder:text-[var(--color-text-faint)]"
            />
            <button
              onClick={onRun}
              disabled={loading}
              className="ml-2 px-4 py-1.5 rounded-full text-xs font-semibold bg-[var(--color-amber)] text-white hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Loading…" : "Analyze"}
            </button>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div
          className={`flex items-center gap-1.5 text-xs ${isDark ? "font-mono" : ""}`}
          style={{ color: isDark ? "var(--color-term-green)" : "var(--color-text-dim)" }}
        >
          <Activity size={12} style={{ color: "var(--color-term-green)" }} />
          <span style={{ color: "var(--color-term-green)" }}>LIVE</span>
        </div>
        <span className={`text-xs tabular ${isDark ? "" : "text-[var(--color-text-dim)]"}`}>
          {timeStr}
        </span>

        <button
          onClick={toggleTheme}
          aria-label="Toggle theme"
          className={`p-1.5 rounded-full border transition-colors ${
            isDark
              ? "border-[var(--color-hairline)] hover:border-[var(--color-amber)] text-[var(--color-text-dim)] hover:text-[var(--color-amber)]"
              : "border-[var(--color-hairline)] hover:border-[var(--color-amber)] text-[var(--color-text-dim)] hover:text-[var(--color-amber)]"
          }`}
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? <Sun size={14} /> : <Moon size={14} />}
        </button>
      </div>
    </header>
  );
};

export default TopBar;
