import React, { useEffect, useState } from "react";
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { api, describeApiError } from "../lib/api";
import { Loader2, TriangleAlert } from "lucide-react";

const GREEN = "#00e676";
const RED = "#ff4757";
const AMBER = "#ffb020";
const CYAN = "#3ddbd9";
const DIM = "#6b6e7b";
const HAIRLINE = "#2a2c35";

const Candle = (props) => {
  const { x, y, width, height, payload } = props;
  const { open, close, high, low } = payload || {};
  if ([open, close, high, low].some((v) => v === undefined || v === null || Number.isNaN(v))) return null;
  if (high === low) return null;

  const isUp = close >= open;
  const color = isUp ? GREEN : RED;

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

const PanelHeader = ({ title, sub }) => (
  <div className="flex items-baseline justify-between mb-2">
    <h3 className="font-mono text-xs tracking-[0.15em] text-[var(--color-text-dim)]">{title}</h3>
    {sub && <span className="font-mono text-[10px] text-[var(--color-text-faint)]">{sub}</span>}
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="bg-[var(--color-panel-raised)] border border-[var(--color-hairline-bright)] px-3 py-2 font-mono text-[11px] text-[var(--color-text-primary)]">
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

const DeepChart = ({ ticker }) => {
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

  if (loading) {
    return (
      <div className="h-96 flex flex-col items-center justify-center gap-2 border border-[var(--color-hairline)] bg-[var(--color-panel)]">
        <Loader2 className="animate-spin text-[var(--color-amber)]" size={28} />
        <span className="font-mono text-xs text-[var(--color-text-dim)] tracking-wide">LOADING {ticker}...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-96 flex flex-col items-center justify-center gap-2 border border-[var(--color-term-red)]/40 bg-[var(--color-panel)]">
        <TriangleAlert className="text-[var(--color-term-red)]" size={28} />
        <span className="font-mono text-xs text-[var(--color-term-red)] text-center max-w-sm px-4">{error}</span>
      </div>
    );
  }

  const priceDomain = ["auto", "auto"];

  return (
    <div className="space-y-4">
      <div className="border border-[var(--color-hairline)] bg-[var(--color-panel)] p-4">
        <PanelHeader title={`PRICE ACTION — ${ticker}`} sub="DAILY · OHLC" />
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={HAIRLINE} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: DIM, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: HAIRLINE }} tickLine={false} minTickGap={40} />
              <YAxis domain={priceDomain} tick={{ fontSize: 10, fill: DIM, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: HAIRLINE }} tickLine={false} width={54} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey={(d) => [d.low ?? 0, d.high ?? 0]} shape={Candle} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="border border-[var(--color-hairline)] bg-[var(--color-panel)] p-4">
        <PanelHeader title="RSI (14)" sub="OVERBOUGHT 70 / OVERSOLD 30" />
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={HAIRLINE} vertical={false} />
              <XAxis dataKey="date" hide />
              <YAxis domain={[0, 100]} ticks={[0, 30, 70, 100]} tick={{ fontSize: 10, fill: DIM, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: HAIRLINE }} tickLine={false} width={54} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={70} stroke={RED} strokeDasharray="3 3" strokeOpacity={0.6} />
              <ReferenceLine y={30} stroke={GREEN} strokeDasharray="3 3" strokeOpacity={0.6} />
              <Line type="monotone" dataKey="RSI_14" stroke={CYAN} dot={false} strokeWidth={1.5} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="border border-[var(--color-hairline)] bg-[var(--color-panel)] p-4">
        <PanelHeader title="MACD (12, 26, 9)" />
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={HAIRLINE} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: DIM, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: HAIRLINE }} tickLine={false} minTickGap={40} />
              <YAxis tick={{ fontSize: 10, fill: DIM, fontFamily: "IBM Plex Mono" }} axisLine={{ stroke: HAIRLINE }} tickLine={false} width={54} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={HAIRLINE} />
              <Bar dataKey="MACDh_12_26_9" isAnimationActive={false}>
                {data.map((d, i) => (
                  <Cell key={i} fill={(d.MACDh_12_26_9 ?? 0) >= 0 ? GREEN : RED} opacity={0.6} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="MACD_12_26_9" stroke={AMBER} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              <Line type="monotone" dataKey="MACDs_12_26_9" stroke={CYAN} dot={false} strokeWidth={1.5} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default DeepChart;
