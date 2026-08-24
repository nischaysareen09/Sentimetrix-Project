import React, { createContext, useContext, useEffect, useState } from "react";

/**
 * ThemeContext — 'light' (default) is the new consumer-fintech look
 * (Tickertape/Investing.com inspired: white cards, blue accent, green/red
 * gain/loss badges, sans-serif). 'dark' is the original amber terminal
 * look we built earlier. Both share the same data-fetching logic; only
 * the visual language differs, branched via `theme` in each component.
 *
 * Persisted in localStorage (this is real app source shipped to the
 * user's own deployment, not a claude.ai artifact — localStorage works
 * fine here, unlike in the sandboxed artifact preview environment).
 */
const ThemeContext = createContext({ theme: "light", toggleTheme: () => {} });

const STORAGE_KEY = "sentimetrix-theme";

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    try {
      return localStorage.getItem(STORAGE_KEY) || "light";
    } catch {
      return "light";
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // localStorage unavailable (private browsing, etc.) — theme just
      // won't persist across reloads, not worth failing over.
    }
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "light" ? "dark" : "light"));

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Colocating the hook with its Provider is the standard React context
// pattern; the following rule is about fast-refresh cache invalidation
// during dev, not a real bug.
// eslint-disable-next-line react-refresh/only-export-components
export const useTheme = () => useContext(ThemeContext);
