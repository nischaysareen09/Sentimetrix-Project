import React, { useEffect, useState } from "react";
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { api, describeApiError } from "../lib/api";
import { Loader2, TriangleAlert } from "lucide-react";
import { useTheme } from "../lib/theme.jsx";

const PALETTE = {
  dark: { green: "#00e676", red: "#ff4757", amber: "#ffb020", cyan: "#3ddbd9", dim: "#6b6e7b", hairline: "#2a2c35" },
  light: { green: "#16a34a", red: "#dc2626", amber: "#2563eb", cyan: "#0891b2", dim: "#667085", hairline: "#e4e7ec" },
};

/**
 * Candlestick body via the "range bar" trick: Bar's dataKey returns
 * [low, high] so recharts gives us real pixel coordinates for that span,
 * then we interpolate the open/close body position within it.
 */
const makeCandle = (colors) => (props) => {
  const { x, y, width, height, payload } = props;
  const { open, close, high, low } = payload || {};
  if ([open, close, high, low].some((v) => v === undefined || v === null || Number.isNaN(v))) return null;
  if (high === low) return null;

  const isUp = close >= open;
  const color = isUp ? colors.green : colors.red;

  const pxPerUnit = height / (high - low);
  const yFor = (val) => y + (high - val) * pxPerUnit;

  const bodyTop = yFor(Math.max(open, close));
  const bodyBottom = yFor(Math.min(open, close));
  const bodyHeight = Math.max(bodyBottom - bodyTop, 1);
  const cx = x + width / 2;
  const bodyWidth = Math.max(width * 0.6, 2);

  return (
    <g>
      <line x1={cx} x2={cx} y1={y} y2={y + height} stroke={color} strokeWidth={1} />
      <rect x={cx - bodyWidth / 2} y={bodyTop} width={bodyWidth} height={bodyHeight} fill={color} opacity={0.9} />
    </g>
  );
};

const DeepChart = ({ ticker }) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const colors = PALETTE[theme];
  const fontFamily = isDark ? "IBM Plex Mono" : "Inter";

  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.get(`/market-data/${ticker}`);
        if (!cancelled) setData(res.data.history || []);
      } catch (err) {
        if (!cancelled) setError(describeApiError(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchData();
    return () => { cancelled = true; };
  }, [ticker]);

  const cardClass = `border border-[var(--color-hairline)] bg-[var(--color-panel)] p-4 rounded-[var(--radius-card)]`;
  const cardStyle = { boxShadow: "var(--shadow-card)" };

  const PanelHeader = ({ title, sub }) => (
    <div className="flex items-baseline justify-between mb-2">
      <h3 className={`text-xs tracking-[0.1em] text-[var(--color-text-dim)] ${isDark ? "font-mono tracking-[0.15em]" : "font-semibold uppercase"}`}>
        {title}
      </h3>
      {sub && <span className="text-[10px] text-[var(--color-text-faint)]">{sub}</span>}
    </div>
  );

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;
    const d = payload[0]?.payload;
    return (
      <div
        className={`px-3 py-2 text-[11px] rounded-md border ${isDark ? "font-mono" : ""}`}
        style={{ backgroundColor: "var(--color-panel-raised)", borderColor: "var(--color-hairline-bright)", color: "var(--color-text-primary)", boxShadow: isDark ? "none" : "0 4px 12px rgba(0,0,0,0.1)" }}
      >
        <div className="text-[var(--color-text-dim)] mb-1">{label}</div>
        {d?.close !== undefined && (
          <>
            <div>O <span className="tabular">{d.open?.toFixed(2)}</span> &nbsp; H <span className="tabular">{d.high?.toFixed(2)}</span></div>
            <div>L <span className="tabular">{d.low?.toFixed(2)}</span> &nbsp; C <span className="tabular">{d.close?.toFixed(2)}</span></div>
          </>
        )}
        {d?.RSI_14 !== undefined && <div>RSI <span className="tabular">{d.RSI_14?.toFixed(1)}</span></div>}
      </div>
    );
  };

  if (loading) {
    return (
      <div className={`h-96 flex flex-col items-center justify-center gap-2 ${cardClass}`} style={cardStyle}>
        <Loader2 className="animate-spin" style={{ color: "var(--color-amber)" }} size={28} />
        <span className={`text-xs text-[var(--color-text-dim)] tracking-wide ${isDark ? "font-mono" : ""}`}>
          Loading {ticker}...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`h-96 flex flex-col items-center justify-center gap-2 border border-[var(--color-term-red)]/40 bg-[var(--color-panel)] rounded-[var(--radius-card)]`}>
        <TriangleAlert style={{ color: "var(--color-term-red)" }} size={28} />
        <span className="text-xs text-center max-w-sm px-4" style={{ color: "var(--color-term-red)" }}>{error}</span>
      </div>
    );
  }

  const Candle = makeCandle(colors);

  return (
    <div className="space-y-4">
      <div className={cardClass} style={cardStyle}>
        <PanelHeader title={`Price Action — ${ticker}`} sub="Daily · OHLC" />
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={colors.hairline} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: colors.dim, fontFamily }} axisLine={{ stroke: colors.hairline }} tickLine={false} minTickGap={40} />
              <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: colors.dim, fontFamily }} axisLine={{ stroke: colors.hairline }} tickLine={false} width={54} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey={(d) => [d.low ?? 0, d.high ?? 0]} shape={Candle} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className={cardClass} style={cardStyle}>
        <PanelHeader title="RSI (14)" sub="Overbought 70 / Oversold 30" />
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={colors.hairline} vertical={false} />
              <XAxis dataKey="date" hide />
              <YAxis domain={[0, 100]} ticks={[0, 30, 70, 100]} tick={{ fontSize: 10, fill: colors.dim, fontFamily }} axisLine={{ stroke: colors.hairline }} tickLine={false} width={54} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={70} stroke={colors.red} strokeDasharray="3 3" strokeOpacity={0.6} />
              <ReferenceLine y={30} stroke={colors.green} strokeDasharray="3 3" strokeOpacity={0.6} />
              <Line type="monotone" dataKey="RSI_14" stroke={colors.cyan} dot={false} strokeWidth={1.5} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className={cardClass} style={cardStyle}>
        <PanelHeader title="MACD (12, 26, 9)" />
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={colors.hairline} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: colors.dim, fontFamily }} axisLine={{ stroke: colors.hairline }} tickLine={false} minTickGap={40} />
              <YAxis tick={{ fontSize: 10, fill: colors.dim, fontFamily }} axisLine={{ stroke: colors.hairline }} tickLine={false} width={54} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={colors.hairline} />
              <Bar dataKey="MACDh_12_26_9" isAnimationActive={false}>
                {data.map((d, i) => (
                  <Cell key={i} fill={(d.MACDh_12_26_9 ?? 0) >= 0 ? colors.green : colors.red} opacity={0.6} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="MACD_12_26_9" stroke={colors.amber} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              <Line type="monotone" dataKey="MACDs_12_26_9" stroke={colors.cyan} dot={false} strokeWidth={1.5} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default DeepChart;
