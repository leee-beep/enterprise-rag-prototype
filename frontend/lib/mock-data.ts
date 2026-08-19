import type { CompanyOption, TopicShortcut } from "@/types/presentation";

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
  "Compare ASUS and Gigabyte's AI strategies.",
  "Compare ASUS and MSI's 2025 gross margins.",
  "How did Gigabyte's operating margin change from 2024 to 2025?",
  "Compare AI server positioning across the three companies.",
];
