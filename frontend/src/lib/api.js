/**
 * api.js — single source of truth for the backend base URL.
 *
 * BUG FIXED: the previous version hardcoded `http://localhost:8000` inside
 * four separate components (App.jsx, DeepChart.jsx, and the
 * /analyze call). That works only on localhost -- any real deployment
 * (Render, Vercel, etc.) where the frontend and backend aren't both on
 * localhost:8000 would have every single API call fail silently to CORS/
 * connection errors, with no single place to fix it.
 *
 * Now: one env var (VITE_API_URL), one axios instance, imported everywhere.
 * Falls back to localhost:8000 for local dev so `npm run dev` still works
 * with zero config.
 *
 * TIMEOUT FIX: `api` (30s) is fine for cheap, fast endpoints (/health,
 * /market-data, /news), but /analyze and /chat load or use heavy
 * transformer models (FinBERT, distilgpt2) on the backend. On Render's
 * free tier, a cold instance can take well over 30s just to wake up and
 * load those models the first time, so every /analyze call was hitting
 * ECONNABORTED before the backend ever got to respond -- even once the
 * backend itself finished successfully seconds later. `slowApi` gives
 * those two endpoints a much longer budget so the request actually has a
 * chance to complete.
 */
import axios from "axios";

export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Use this instance for /analyze and /chat specifically -- the two routes
// that can trigger a slow, cold heavy-model load on the backend.
export const slowApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes -- enough headroom for a cold Render wake + model load
});

export function describeApiError(err) {
  if (err?.code === "ECONNABORTED") return "Request timed out. The backend may be cold-starting -- try again in a few seconds.";
  if (err?.response?.data?.detail) return err.response.data.detail;
  if (err?.response?.status === 404) return "Not found.";
  if (err?.message === "Network Error") return "Can't reach the backend. Is it running and is VITE_API_URL set correctly?";
  return err?.message || "Something went wrong.";
}
