"use client";
import { useState } from "react";
import type { HistoryEntry } from "@/components/layout/sidebar";

const HISTORY_KEY = "competitor-intelligence-session-history";

export function useSessionHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY) ?? "[]") as HistoryEntry[]; }
    catch { return []; }
  });
  function addHistory(question: string) {
    const entry = { question, timestamp: new Date().toISOString(), status: "preview" };
    setHistory((current) => {
      const next = [entry, ...current].slice(0, 8);
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      return next;
    });
  }
  return { history, addHistory };
}
