import React, { useState, useEffect, useRef } from "react";
import { api, describeApiError } from "../lib/api";
import { Send, Terminal, MessageCircle, User } from "lucide-react";
import { useTheme } from "../lib/theme.jsx";

const AIAnalyst = ({ ticker, context, signal }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const HeaderIcon = isDark ? Terminal : MessageCircle;

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    setMessages([
      { role: "assistant", content: `Ready. Ask about ${ticker}'s technicals, the RAG rules that fired, or the signal above.` },
    ]);
  }, [ticker]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    const userMsg = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await api.post("/chat", {
        ticker,
        query: trimmed,
        context: context || "No news context provided.",
        signal: signal ?? 1,
      });
      setMessages((prev) => [...prev, { role: "assistant", content: res.data.response }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${describeApiError(err)}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[var(--color-panel)] border-l border-[var(--color-hairline)]">
      <div className="px-4 py-3 border-b border-[var(--color-hairline)] flex items-center gap-2">
        <HeaderIcon size={15} style={{ color: "var(--color-amber)" }} />
        <h2 className={`text-sm font-semibold text-[var(--color-text-primary)] ${isDark ? "font-mono text-xs tracking-[0.15em]" : ""}`}>
          {isDark ? "AI ANALYST" : "AI Analyst"}
        </h2>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto term-scroll p-4 space-y-3">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] px-3 py-2 text-xs leading-relaxed border rounded-[var(--radius-card)] ${
                isDark ? "font-mono" : ""
              } ${
                m.role === "user"
                  ? "border-[var(--color-amber)]/40 text-[var(--color-text-primary)]"
                  : "border-[var(--color-hairline)] text-[var(--color-text-primary)] bg-[var(--color-panel-raised)]"
              }`}
              style={{ boxShadow: isDark ? "none" : "var(--shadow-card)" }}
            >
              <div className="flex items-center gap-1.5 mb-1 text-[10px] text-[var(--color-text-dim)] tracking-wide">
                {m.role === "user" ? <User size={10} /> : <HeaderIcon size={10} style={{ color: "var(--color-amber)" }} />}
                {m.role === "user" ? "You" : "Analyst"}
              </div>
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-[11px] text-[var(--color-text-dim)] flex items-center gap-1">
            {isDark && <span className="blink-cursor" style={{ color: "var(--color-amber)" }}>▮</span>} analyzing...
          </div>
        )}
      </div>

      <div className="p-3 border-t border-[var(--color-hairline)]">
        <div className={`flex items-center gap-2 border border-[var(--color-hairline)] focus-within:border-[var(--color-amber)] px-2.5 py-1.5 transition-colors rounded-[var(--radius-card)]`}>
          <input
            className={`flex-1 bg-transparent outline-none text-xs text-[var(--color-text-primary)] placeholder:text-[var(--color-text-faint)] ${isDark ? "font-mono" : ""}`}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about MACD, trend, RSI..."
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            style={{ color: input.trim() && !loading ? "var(--color-amber)" : "var(--color-text-faint)" }}
            className="transition-colors"
            aria-label="Send"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default AIAnalyst;
