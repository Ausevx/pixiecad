import { useEffect, useState } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   Theme.

   Three states originally, now a registry of themes. "system" is the default
   and is genuinely distinct from picking a specific theme: it keeps following
   the OS afterwards. Collapsing it into a boolean would silently freeze
   whichever mode the user happened to be in the first time they visited.
   ───────────────────────────────────────────────────────────────────────── */

export type ThemeId = "dark" | "light" | "midnight" | "nord" | "sepia" | "contrast";
export type ThemeChoice = ThemeId | "system";
export type Resolved = "light" | "dark";

export interface ThemeMeta {
  id: ThemeId;
  /** Shown in the picker. */
  label: string;
  /** Which of the two underlying modes this theme is built on. Drives the
   *  resolved value and anything that still needs a binary light/dark. */
  base: Resolved;
  /** A representative colour, used for the picker's swatch. Must be the
   *  theme's own --px-accent value. */
  swatch: string;
}

/* Each swatch duplicates that theme's --px-accent from styles.css. The
   duplication is deliberate: the picker paints its swatches into inline styles
   before the theme is applied, so it cannot read the variable off :root — that
   would only ever yield the *current* theme's accent. Changing an accent means
   changing it in both places. */
/** Dark is named separately because it is the fallback for every unknown id,
 *  and indexing THEMES for it would be an unchecked access under this
 *  tsconfig. */
const DARK: ThemeMeta = { id: "dark", label: "amber dark", base: "dark", swatch: "#ffb230" };

export const THEMES: readonly ThemeMeta[] = [
  DARK,
  { id: "light", label: "paper", base: "light", swatch: "#8a5200" },
  { id: "midnight", label: "midnight", base: "dark", swatch: "#38bdf8" },
  { id: "nord", label: "nord", base: "dark", swatch: "#88c0d0" },
  { id: "sepia", label: "sepia", base: "light", swatch: "#85440e" },
  { id: "contrast", label: "high contrast", base: "dark", swatch: "#ffee00" },
];

/** Look-up that cannot fail, so the picker never has to handle a missing row.
 *  An unknown id means a stale stored value, and dark is the design's home. */
export function themeMeta(id: ThemeId): ThemeMeta {
  return THEMES.find((t) => t.id === id) ?? DARK;
}

const KEY = "pixiecad.theme";
const listeners = new Set<(t: ThemeChoice) => void>();

/** Narrows a raw stored string rather than casting it, so a theme removed in a
 *  later build cannot smuggle a dead id back into the running app. */
const isThemeId = (v: string | null): v is ThemeId => THEMES.some((t) => t.id === v);

function read(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return isThemeId(v) ? v : "system";
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

export const resolve = (c: ThemeChoice): Resolved => {
  if (c === "system") return systemPrefersLight() ? "light" : "dark";
  return themeMeta(c).base;
};

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
  };
}
