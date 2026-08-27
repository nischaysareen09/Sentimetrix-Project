import React, { useEffect, useState } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";
import { api } from "../lib/api";

const REFRESH_MS = 5 * 60 * 1000; // matches backend cache TTL, no point polling faster

/**
 * TickerTape -- continuously scrolling strip of live quotes across the
 * very top of the page, matching the NSE/Tickertape.in convention of a
 * ticker tape above the main nav. Pure CSS animation (translateX loop),
 * not JS-driven, so it costs nothing at runtime and never jitters.
 *
 * Seamless-loop trick: render the quote list TWICE back-to-back inside a
 * flex row, then animate translateX(0 -> -50%). Since the second copy is
 * an exact duplicate of the first, the moment the first copy has fully
 * scrolled off-screen the second copy is in exactly the position the
 * first one started in -- no visible seam or jump.
 */
const TickerTape = () => {
  const [quotes, setQuotes] = useState([]);

  useEffect(() => {
    let cancelled = false;

    const fetchTape = async () => {
      try {
        const res = await api.get("/tape");
        if (!cancelled) setQuotes(res.data.tape || []);
      } catch {
        // Decorative feature -- fail silently, just don't render the tape
        // rather than showing an error banner for something non-essential.
      }
    };

    fetchTape();
    const id = setInterval(fetchTape, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (quotes.length === 0) return null;

  const renderQuotes = (keyPrefix) =>
    quotes.map((q, i) => {
      const isUp = q.change_pct >= 0;
      const Icon = isUp ? TrendingUp : TrendingDown;
      const color = isUp ? "var(--color-term-green)" : "var(--color-term-red)";
      return (
        <div key={`${keyPrefix}-${i}`} className="flex items-center gap-2 px-5 flex-shrink-0 font-mono text-xs">
          <span className="text-[var(--color-text-primary)] font-semibold tracking-wide">{q.display}</span>
          <span className="text-[var(--color-text-dim)] tabular">{q.price.toLocaleString()}</span>
          <span className="flex items-center gap-0.5 tabular" style={{ color }}>
            <Icon size={11} />
            {isUp ? "+" : ""}{q.change_pct.toFixed(2)}%
          </span>
        </div>
      );
    });

  return (
    <div className="border-b border-[var(--color-hairline)] bg-[var(--color-panel)] overflow-hidden py-2">
      <div className="ticker-tape-track flex w-max">
        {renderQuotes("a")}
        {renderQuotes("b")}
      </div>
    </div>
  );
};

export default TickerTape;