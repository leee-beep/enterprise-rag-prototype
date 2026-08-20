import WorkspaceClient from "./workspace-client";
import { LocaleProvider } from "@/lib/i18n-context";

export default function Home() { return <LocaleProvider><WorkspaceClient /></LocaleProvider>; }
