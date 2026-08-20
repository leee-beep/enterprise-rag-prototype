"use client";

import { createContext, useContext, useMemo, useSyncExternalStore, type ReactNode } from "react";
import { DEFAULT_LOCALE, LOCALE_STORAGE_KEY, normalizeStoredLocale, translate, type Locale } from "./i18n";

const LocaleContext = createContext<{
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: Parameters<typeof translate>[1]) => string;
} | null>(null);

const LOCALE_EVENT = "competitor-intelligence-locale-change";

function subscribe(callback: () => void) {
  window.addEventListener(LOCALE_EVENT, callback);
  return () => window.removeEventListener(LOCALE_EVENT, callback);
}

function browserLocale(): Locale {
  return normalizeStoredLocale(window.localStorage.getItem(LOCALE_STORAGE_KEY));
}

export function LocaleProvider({ children }: { children: ReactNode }) {
  const locale = useSyncExternalStore(subscribe, browserLocale, () => DEFAULT_LOCALE);
  const value = useMemo(() => ({
    locale,
    setLocale(next: Locale) {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, next);
      window.dispatchEvent(new Event(LOCALE_EVENT));
    },
    t: (key: Parameters<typeof translate>[1]) => translate(locale, key),
  }), [locale]);
  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}

export function useI18n() {
  const value = useContext(LocaleContext);
  if (!value) throw new Error("useI18n must be used inside LocaleProvider");
  return value;
}
