export type Locale = "zh-TW" | "en";

export const DEFAULT_LOCALE: Locale = "zh-TW";
export const LOCALE_STORAGE_KEY = "competitor-intelligence-ui-locale";

const en = {
  appName: "Competitor Intelligence", localEnvironment: "Local research environment", privateLocal: "Private · Local",
  ready: "Ready", connecting: "Connecting", unavailable: "Unavailable", unknown: "Unknown", retry: "Retry",
  companies: "Companies", themes: "Research themes", recentQueries: "Recent queries", clear: "Clear",
  localHistory: "Questions stay in this browser session only.", researchNavigation: "Research navigation",
  closeNavigation: "Close navigation", openNavigation: "Open research navigation", openEvidence: "Open evidence panel",
  researchQuestion: "Research question", localAnalysis: "Local analysis", queryPlaceholder: "Ask a comparative strategy or financial question…",
  context: "Context", analyze: "Analyze", analyzing: "Analyzing", evidenceDecisionSupport: "Evidence-backed decision support",
  heroTitle: "Research competitors. Verify every conclusion.", heroDescription: "Compare strategy and validated financial performance across ASUS, Gigabyte, and MSI using local annual-report evidence.",
  researchScope: "Research scope", emptyTitle: "Grounded comparison, ready when you are", emptyDescription: "Ask about strategy, positioning, risk, or verified financial performance. Every live conclusion stays connected to structured evidence.",
  analysis: "Analysis", authority: "Authority", strategyFinancialsChange: "Strategy · Financials · Change", annualReportsFacts: "Annual reports · Validated facts",
  analysisProgress: "Analysis in progress", buildingBrief: "Building your evidence-backed brief", phaseSearch: "Searching trusted evidence", phaseFinancial: "Preparing financial analysis", phaseGenerate: "Generating a grounded response", phaseProvenance: "Organizing provenance", loadingNote: "These indicators describe the experience, not measured backend phases.",
  liveAnalysis: "Live analysis", coverageNote: "Coverage note", executiveComparison: "Executive comparison", comparisonTitle: "Verified company evidence at a glance",
  comparisonDisclosure: "The current API verifies company-level evidence coverage, but does not yet provide authoritative structured strategy summaries. No comparison claims are inferred from prose here.",
  keyTakeaway: "Key takeaway", takeawayTitle: "A grounded cross-company conclusion requires a structured synthesis contract.", takeawayBody: "The current response proves evidence coverage for each company, but does not safely isolate one authoritative strategic difference. The detailed analysis is preserved below without manufacturing a comparison.",
  supportingNarrative: "Supporting narrative", detailedAnalysis: "Detailed grounded analysis", deterministicAuthority: "Deterministic authority", financialComparison: "Financial comparison", financialDisclosure: "Values and ranks below are displayed exactly as supplied by the validated backend contract.",
  structuredProvenance: "Structured provenance", sourcesEvidence: "Sources and evidence", evidence: "Evidence", sources: "Sources", source: "Source", sourceDetailsUnavailable: "Source details not available", localAI: "Local AI",
  verified: "Verified", qualitativeItems: "qualitative items", evidenceLabel: "Evidence", page: "Page", pages: "Pages", document: "Document", metric: "Metric", company: "Company", year: "Year", role: "Role", rank: "Rank", notAvailable: "Not available",
  financial: "Financial", qualitative: "Qualitative", noEvidence: "No evidence yet", evidenceEmpty: "Structured citations and financial provenance appear after analysis.", previousEvidence: "Previous evidence", nextEvidence: "Next evidence", closeEvidence: "Close evidence panel",
  completed: "Completed", partial: "Partial", ambiguous: "Needs refinement", unsupported: "Unsupported", insufficient: "Insufficient evidence",
  completedDetail: "Grounded analysis is ready for review.", partialDetail: "The available result is usable, with evidence limitations noted below.", ambiguousDetail: "Make the company, year, or analysis goal more specific.", unsupportedDetail: "This request is outside the workspace's current analysis capabilities.", insufficientDetail: "There is not enough trusted evidence for a grounded conclusion.",
  serviceUnavailable: "Local service unavailable", serviceUnavailableDetail: "Check the local API and model service, then retry when ready.", interrupted: "Analysis interrupted", interruptedDetail: "The request could not be completed. Your question has been retained.",
  localeChinese: "中文", localeEnglish: "EN", localeSwitcher: "Interface language",
  reasonQualitativeUnavailable: "Trusted qualitative evidence is unavailable for part of this request.", reasonCompany: "Name the company or companies you want to analyze.", reasonYear: "Add a fiscal year to make the request precise.", reasonCompanies: "Choose at least two companies for a comparison.", reasonMetric: "Specify the financial metric you want to compare.", reasonIntent: "Clarify the type of competitor analysis you need.", reasonFallback: "Additional trusted evidence or a more specific question may be required.",
  verifiedEvidenceCoverage: "Verified Evidence Coverage", strategyProfiles: "Company strategy profiles", comparisonDimensions: "Comparison dimensions", missingCompanies: "Missing company evidence",
} as const;

type TranslationKey = keyof typeof en;
const zhTW: Record<TranslationKey, string> = {
  appName: "競品情報分析", localEnvironment: "本機研究環境", privateLocal: "私有 · 本機",
  ready: "已就緒", connecting: "連線中", unavailable: "無法使用", unknown: "狀態未知", retry: "重試",
  companies: "公司", themes: "研究主題", recentQueries: "最近查詢", clear: "清除",
  localHistory: "問題只保留於目前瀏覽器工作階段。", researchNavigation: "研究導覽", closeNavigation: "關閉導覽", openNavigation: "開啟研究導覽", openEvidence: "開啟證據面板",
  researchQuestion: "研究問題", localAnalysis: "本機分析", queryPlaceholder: "輸入策略或財務比較問題…", context: "分析範圍", analyze: "開始分析", analyzing: "分析中",
  evidenceDecisionSupport: "有據可查的決策支援", heroTitle: "洞察競爭態勢，驗證每項結論。", heroDescription: "運用本機年報證據，比較 ASUS、Gigabyte 與 MSI 的策略及經驗證財務表現。",
  researchScope: "研究範圍", emptyTitle: "準備好進行有依據的競品比較", emptyDescription: "可詢問策略、定位、風險或經驗證的財務表現；每項即時結論都連結至結構化證據。",
  analysis: "分析面向", authority: "資料依據", strategyFinancialsChange: "策略 · 財務 · 變化", annualReportsFacts: "年度報告 · 驗證事實",
  analysisProgress: "分析進行中", buildingBrief: "正在建立有據可查的分析摘要", phaseSearch: "搜尋可信證據", phaseFinancial: "準備財務分析", phaseGenerate: "產生有依據的回應", phaseProvenance: "整理證據來源", loadingNote: "此處顯示使用體驗階段，不代表實際後端量測。",
  liveAnalysis: "即時分析", coverageNote: "涵蓋範圍說明", executiveComparison: "競品比較摘要", comparisonTitle: "各公司已驗證證據一覽",
  comparisonDisclosure: "目前 API 可驗證公司層級的證據涵蓋，但尚未提供權威的結構化策略摘要；此處不會由一般敘述推導比較主張。",
  keyTakeaway: "核心結論", takeawayTitle: "完整跨公司結論仍需要結構化綜合分析契約。", takeawayBody: "目前回應能證明各公司的證據涵蓋，但無法安全分離出單一權威策略差異；下方保留詳細分析，且不製造比較結論。",
  supportingNarrative: "補充敘述", detailedAnalysis: "詳細分析", deterministicAuthority: "確定性資料依據", financialComparison: "財務比較", financialDisclosure: "下列數值與排名完全依照已驗證的後端契約顯示。",
  structuredProvenance: "結構化來源追溯", sourcesEvidence: "資料來源與證據", evidence: "證據來源", sources: "資料來源", source: "來源", sourceDetailsUnavailable: "無可用來源資訊", localAI: "本機 AI",
  verified: "已驗證", qualitativeItems: "筆定性證據", evidenceLabel: "證據", page: "頁碼", pages: "頁碼", document: "文件", metric: "指標", company: "公司", year: "年度", role: "角色", rank: "排名", notAvailable: "無資料",
  financial: "財務", qualitative: "定性", noEvidence: "尚無證據", evidenceEmpty: "完成分析後將顯示結構化引用與財務來源。", previousEvidence: "上一筆證據", nextEvidence: "下一筆證據", closeEvidence: "關閉證據面板",
  completed: "分析完成", partial: "部分完成", ambiguous: "需要釐清", unsupported: "尚未支援", insufficient: "證據不足",
  completedDetail: "有依據的分析已可供檢閱。", partialDetail: "結果可使用，但仍有下方所列的證據限制。", ambiguousDetail: "請更明確指定公司、年度或分析目的。", unsupportedDetail: "此要求超出目前工作區的分析能力。", insufficientDetail: "目前沒有足夠可信證據支持結論。",
  serviceUnavailable: "本機服務無法使用", serviceUnavailableDetail: "請檢查本機 API 與模型服務後再重試。", interrupted: "分析中斷", interruptedDetail: "無法完成本次要求，您的問題仍保留在輸入區。",
  localeChinese: "中文", localeEnglish: "EN", localeSwitcher: "介面語言",
  reasonQualitativeUnavailable: "此要求的部分內容缺少可信定性證據。", reasonCompany: "請指定要分析的公司。", reasonYear: "請加入會計年度以明確化要求。", reasonCompanies: "比較至少需要選擇兩家公司。", reasonMetric: "請指定要比較的財務指標。", reasonIntent: "請說明需要的競品分析類型。", reasonFallback: "可能需要更多可信證據或更具體的問題。",
  verifiedEvidenceCoverage: "已驗證證據範圍", strategyProfiles: "公司策略摘要", comparisonDimensions: "比較面向", missingCompanies: "缺少公司證據",
};

const dictionaries = { en, "zh-TW": zhTW } as const;

export function translate(locale: Locale, key: TranslationKey): string {
  return dictionaries[locale]?.[key] ?? en[key];
}

export function normalizeStoredLocale(value: string | null): Locale {
  return value === "en" ? "en" : DEFAULT_LOCALE;
}

export const localizedPrompts: Record<Locale, string[]> = {
  en: ["Compare ASUS and Gigabyte's AI strategies.", "Compare ASUS and MSI's 2025 gross margins.", "How did Gigabyte's operating margin change from 2024 to 2025?", "Compare AI server positioning across the three companies."],
  "zh-TW": ["比較 ASUS 與 Gigabyte 的 AI 策略。", "比較 ASUS 與 MSI 的 2025 年毛利率。", "Gigabyte 的營業利益率從 2024 到 2025 年如何變化？", "比較三家公司在 AI 伺服器市場的定位。"],
};

export const localizedTopics: Record<Locale, Array<{ label: string; query: string }>> = {
  en: [{ label: "AI strategy", query: "Compare how the selected companies describe their AI strategy." }, { label: "AI infrastructure", query: "Compare the selected companies' AI infrastructure positioning." }, { label: "AI servers", query: "Compare the selected companies' AI server strategies." }, { label: "AI PCs", query: "Compare the selected companies' AI PC strategies." }, { label: "Financial performance", query: "Compare 2025 operating margin trends across the selected companies." }, { label: "Risk factors", query: "What material strategic risks do the selected companies report?" }],
  "zh-TW": [{ label: "AI 策略", query: "比較所選公司如何描述其 AI 策略。" }, { label: "AI 基礎設施", query: "比較所選公司的 AI 基礎設施定位。" }, { label: "AI 伺服器", query: "比較所選公司的 AI 伺服器策略。" }, { label: "AI PC", query: "比較所選公司的 AI PC 策略。" }, { label: "財務表現", query: "比較所選公司 2025 年的營業利益率趨勢。" }, { label: "風險因素", query: "所選公司揭露了哪些重大策略風險？" }],
};
