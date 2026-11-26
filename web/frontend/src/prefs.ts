// Small shared helpers: persisted preferences and the mobile breakpoint.
import { useEffect, useState } from "react";

export function readPref<T>(k: string, d: T): T {
  try { const v = localStorage.getItem(k); return v ? (JSON.parse(v) as T) : d; } catch { return d; }
}
export function writePref(k: string, v: unknown) {
  try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ }
}

/** Below this width the sidebar becomes a drawer and the settings panel a bottom sheet (keep in sync with styles.css). */
export const MOBILE_QUERY = "(max-width: 760px)";

export const isMobileNow = () => typeof window !== "undefined" && window.matchMedia(MOBILE_QUERY).matches;

export function useIsMobile(): boolean {
  const [m, setM] = useState(isMobileNow);
  useEffect(() => {
    const mq = window.matchMedia(MOBILE_QUERY);
    const on = () => setM(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return m;
}
