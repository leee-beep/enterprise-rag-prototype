"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SyntheticChartDatum } from "@/types/presentation";

export function MarginChart({ data }: { data: SyntheticChartDatum[] }) {
  return (
    <div className="chart-card">
      <div><span>Synthetic preview</span><h4>Operating margin comparison</h4></div>
      <div
        className="chart-frame"
        role="img"
        aria-label="Synthetic operating margin comparison chart"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke="#253532" />
            <XAxis dataKey="company" tick={{ fill: "#91a49f", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 8]} tick={{ fill: "#657a75", fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip cursor={{ fill: "rgba(154,242,181,.05)" }} contentStyle={{ background: "#101d1b", border: "1px solid #345047", borderRadius: 8 }} />
            <Bar dataKey="value" fill="#62d78a" radius={[5, 5, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
