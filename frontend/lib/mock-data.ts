import type {
  CompanyOption,
  SyntheticChartDatum,
  TopicShortcut,
} from "@/types/presentation";

export const companies: CompanyOption[] = [
  { id: "asus", name: "ASUS", ticker: "2357" },
  { id: "gigabyte", name: "Gigabyte", ticker: "2376" },
  { id: "msi", name: "MSI", ticker: "2377" },
];

export const topics: TopicShortcut[] = [
  { label: "AI strategy", query: "Compare how the selected companies describe their AI strategy." },
  { label: "AI infrastructure", query: "Compare the selected companies' AI infrastructure positioning." },
  { label: "AI servers", query: "Compare the selected companies' AI server strategies." },
  { label: "AI PCs", query: "Compare the selected companies' AI PC strategies." },
  { label: "Financial performance", query: "Compare 2025 operating margin trends across the selected companies." },
  { label: "Margins", query: "Compare gross, operating, and net margins for the selected companies." },
  { label: "Growth", query: "Compare revenue growth for the selected companies." },
  { label: "Risk factors", query: "What material strategic risks do the selected companies report?" },
];

export const previewPrompts = [
  "Compare 2025 operating margin trends",
  "What strategic risks do the reports discuss?",
  "Summarize each company's AI positioning",
];

// Explicitly synthetic preview content. No private document text or real facts.
// Isolated showcase-only numbers; real API financial strings are never parsed.
export const syntheticMarginChartData: SyntheticChartDatum[] = [
  { company: "ASUS", value: 6.4 },
  { company: "Gigabyte", value: 7.1 },
  { company: "MSI", value: 6.4 },
];
