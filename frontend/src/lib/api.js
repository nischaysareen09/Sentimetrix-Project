/**
 * api.js — single source of truth for the backend base URL.
 *
 * BUG FIXED: the previous version hardcoded `http://localhost:8000` inside
 * four separate components (App.jsx, DeepChart.jsx, AIAnalyst.jsx, and the
 * /analyze call). That works only on localhost — any real deployment
 * (Render, Vercel, etc.) where the frontend and backend aren't both on
 * localhost:8000 would have every single API call fail silently to CORS/
 * connection errors, with no single place to fix it.
 *
 * Now: one env var (VITE_API_URL), one axios instance, imported everywhere.
 * Falls back to localhost:8000 for local dev so `npm run dev` still works
 * with zero config.
 */
import axios from "axios";

export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Central error normalizer so every component gets a consistent, human
// -readable message instead of each one guessing at err.message shapes.
export function describeApiError(err) {
  if (err?.code === "ECONNABORTED") return "Request timed out. The backend may be cold-starting — try again in a few seconds.";
  if (err?.response?.data?.detail) return err.response.data.detail;
  if (err?.response?.status === 404) return "Not found.";
  if (err?.message === "Network Error") return "Can't reach the backend. Is it running and is VITE_API_URL set correctly?";
  return err?.message || "Something went wrong.";
}
