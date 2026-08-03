import { useEffect, useState } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   Theme.

   Three states, not two. "system" is the default and is genuinely distinct
   from picking dark: it keeps following the OS afterwards. Collapsing it into
   a boolean would silently freeze whichever mode the user happened to be in
   the first time they visited.
   ───────────────────────────────────────────────────────────────────────── */

export type ThemeChoice = "system" | "light" | "dark";
export type Resolved = "light" | "dark";

const KEY = "pixiecad.theme";
const listeners = new Set<(t: ThemeChoice) => void>();

function read(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    // Private browsing and some hardened configurations throw on access
    // rather than returning null. Falling back to system is correct.
    return "system";
  }
}

let choice: ThemeChoice = typeof window === "undefined" ? "system" : read();

export const systemPrefersLight = (): boolean =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-color-scheme: light)").matches;

export const resolve = (c: ThemeChoice): Resolved =>
  c === "system" ? (systemPrefersLight() ? "light" : "dark") : c;

/** Stamp the choice on <html>. The stylesheet keys off [data-theme], and its
 *  absence is what lets the prefers-color-scheme media query take over — so
 *  "system" must REMOVE the attribute rather than write a resolved value. */
function apply(c: ThemeChoice): void {
  const root = document.documentElement;
  if (c === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", c);
}

export function setTheme(c: ThemeChoice): void {
  choice = c;
  try {
    if (c === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, c);
  } catch {
    /* the choice still applies for this session */
  }
  apply(c);
  listeners.forEach((l) => l(c));
}

/** Called once before React mounts, so the first paint is already correct.
 *  Doing this in an effect instead would flash the wrong theme. */
export function initTheme(): void {
  apply(choice);
}

export function useTheme(): {
  choice: ThemeChoice;
  resolved: Resolved;
  setTheme: (c: ThemeChoice) => void;
  cycle: () => void;
} {
  const [local, setLocal] = useState<ThemeChoice>(choice);
  const [, force] = useState(0);

  useEffect(() => {
    listeners.add(setLocal);
    // While on "system", an OS change must repaint immediately — the CSS
    // handles the colours, but anything reading the resolved value in JS
    // (the background canvas, the logo) needs to be told.
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const onSystem = () => force((n) => n + 1);
    mq.addEventListener("change", onSystem);
    return () => {
      listeners.delete(setLocal);
      mq.removeEventListener("change", onSystem);
    };
  }, []);

  return {
    choice: local,
    resolved: resolve(local),
    setTheme,
    cycle: () => setTheme(local === "system" ? "dark" : local === "dark" ? "light" : "system"),
  };
}
