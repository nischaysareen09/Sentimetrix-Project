import React from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useTheme } from "../lib/theme.jsx";

const SIGNAL_CONFIG = {
  0: { label: "STRONG SELL", lightLabel: "Sell Signal", color: "var(--color-term-red)", glow: "rgba(255,71,87,0.35)", bar: "bg-[var(--color-term-red)]", Icon: TrendingDown },
  1: { label: "HOLD / NEUTRAL", lightLabel: "Hold", color: "var(--color-amber)", glow: "rgba(255,176,32,0.35)", bar: "bg-[var(--color-amber)]", Icon: Minus },
  2: { label: "STRONG BUY", lightLabel: "Buy Signal", color: "var(--color-term-green)", glow: "rgba(0,230,118,0.35)", bar: "bg-[var(--color-term-green)]", Icon: TrendingUp },
};

/**
 * SignalReadout — the two themes diverge the most here:
 * - dark: amber CRT-glow scanline readout (original design)
 * - light: a clean "quote card" like a Tickertape/Investing.com stock
 *   header — big ticker name, a colored pill badge for the signal, and a
 *   confidence meter styled like a normal progress bar, not a glow effect
 */
const SignalReadout = ({ signalClass, confidence, ticker }) => {
  const { theme } = useTheme();
  const cfg = SIGNAL_CONFIG[signalClass] ?? SIGNAL_CONFIG[1];
  const Icon = cfg.Icon;

  if (theme === "dark") {
    return (
      <div
        className="scanlines relative border border-[var(--color-hairline)] bg-[var(--color-panel)] px-6 py-6 overflow-hidden"
        style={{ boxShadow: `inset 0 0 60px -20px ${cfg.glow}` }}
      >
        <div className="flex items-center justify-between mb-3">
          <span className="font-mono text-[11px] tracking-[0.2em] text-[var(--color-text-dim)]">
            INTELLIGENCE SIGNAL — {ticker}
          </span>
          <span className="font-mono text-[11px] tracking-widest text-[var(--color-text-dim)]">TCN v2</span>
        </div>

        <div
          className="font-mono text-4xl md:text-5xl font-bold tracking-tight glow-amber"
          style={{ color: cfg.color, textShadow: `0 0 24px ${cfg.glow}, 0 0 4px ${cfg.glow}` }}
        >
          {cfg.label}
        </div>

        <div className="mt-4 flex items-center gap-3">
          <span className="font-mono text-xs text-[var(--color-text-dim)] tracking-wide">CONFIDENCE</span>
          <div className="flex-1 h-1.5 bg-[var(--color-hairline)] max-w-xs">
            <div className={`h-full ${cfg.bar}`} style={{ width: `${Math.min(Math.max(confidence, 0), 100)}%` }} />
          </div>
          <span className="font-mono text-sm tabular" style={{ color: cfg.color }}>
            {confidence.toFixed(1)}%
          </span>
        </div>
      </div>
    );
  }

  // Light theme: clean quote card
  return (
    <div
      className="rounded-[var(--radius-card)] border border-[var(--color-hairline)] bg-white px-6 py-5"
      style={{ boxShadow: "var(--shadow-card)" }}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs text-[var(--color-text-dim)] mb-1">AI Signal for</div>
          <div className="text-2xl font-bold text-[var(--color-text-primary)]">{ticker}</div>
        </div>
        <div
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold"
          style={{ backgroundColor: cfg.glow, color: cfg.color }}
        >
          <Icon size={15} />
          {cfg.lightLabel}
        </div>
      </div>

      <div className="mt-5">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-[var(--color-text-dim)]">Model confidence</span>
          <span className="text-sm font-semibold tabular" style={{ color: cfg.color }}>
            {confidence.toFixed(1)}%
          </span>
        </div>
        <div className="h-2 rounded-full bg-[var(--color-hairline)] overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${Math.min(Math.max(confidence, 0), 100)}%`, backgroundColor: cfg.color }}
          />
        </div>
      </div>
    </div>
  );
};

export default SignalReadout;
