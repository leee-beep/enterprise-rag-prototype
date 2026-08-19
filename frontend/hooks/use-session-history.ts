"use client";
import { useSyncExternalStore } from "react";
import type { HistoryEntry } from "@/components/layout/sidebar";

const HISTORY_KEY = "competitor-intelligence-session-history";
const EMPTY_HISTORY: HistoryEntry[] = [];
const listeners = new Set<() => void>();
let cachedRaw: string | null = null;
let cachedHistory: HistoryEntry[] = EMPTY_HISTORY;

function snapshot(): HistoryEntry[] {
  const raw = sessionStorage.getItem(HISTORY_KEY) ?? "[]";
  if (raw === cachedRaw) return cachedHistory;
  cachedRaw = raw;
  try { cachedHistory = JSON.parse(raw) as HistoryEntry[]; }
  catch { cachedHistory = EMPTY_HISTORY; }
  return cachedHistory;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useSessionHistory() {
  const history = useSyncExternalStore(subscribe, snapshot, () => EMPTY_HISTORY);
  function addHistory(question: string, status: string) {
    const entry = { question, timestamp: new Date().toISOString(), status };
    const next = [entry, ...snapshot()].slice(0, 8);
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(next));
    cachedRaw = null;
    listeners.forEach((listener) => listener());
  }
  return { history, addHistory };
}
