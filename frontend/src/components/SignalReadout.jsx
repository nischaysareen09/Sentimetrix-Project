import React from "react";

const SIGNAL_CONFIG = {
  0: { label: "STRONG SELL", color: "var(--color-term-red)", glow: "rgba(255,71,87,0.35)", bar: "bg-[var(--color-term-red)]" },
  1: { label: "HOLD / NEUTRAL", color: "var(--color-amber)", glow: "rgba(255,176,32,0.35)", bar: "bg-[var(--color-amber)]" },
  2: { label: "STRONG BUY", color: "var(--color-term-green)", glow: "rgba(0,230,118,0.35)", bar: "bg-[var(--color-term-green)]" },
};

const SignalReadout = ({ signalClass, confidence, ticker }) => {
  const cfg = SIGNAL_CONFIG[signalClass] ?? SIGNAL_CONFIG[1];

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
          <div
            className={`h-full ${cfg.bar}`}
            style={{ width: `${Math.min(Math.max(confidence, 0), 100)}%` }}
          />
        </div>
        <span className="font-mono text-sm tabular" style={{ color: cfg.color }}>
          {confidence.toFixed(1)}%
        </span>
      </div>
    </div>
  );
};

export default SignalReadout;
