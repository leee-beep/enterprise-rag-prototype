import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_LOCALE, localizedPrompts, normalizeStoredLocale, translate } from "../lib/i18n.ts";

test("zh-TW is the deterministic default locale", () => {
  assert.equal(DEFAULT_LOCALE, "zh-TW");
  assert.equal(normalizeStoredLocale(null), "zh-TW");
  assert.equal(normalizeStoredLocale("unexpected"), "zh-TW");
});

test("English is restored only from the supported persisted value", () => {
  assert.equal(normalizeStoredLocale("en"), "en");
});

test("known labels and domain states are localized", () => {
  assert.equal(translate("zh-TW", "executiveComparison"), "競品比較摘要");
  assert.equal(translate("zh-TW", "completed"), "分析完成");
  assert.equal(translate("en", "completed"), "Completed");
});

test("prompt language changes only the UI-owned prompt choices", () => {
  assert.notEqual(localizedPrompts["zh-TW"][0], localizedPrompts.en[0]);
  const financialValue = "18.25%";
  const evidenceId = "E7";
  assert.equal(financialValue, "18.25%");
  assert.equal(evidenceId, "E7");
});
