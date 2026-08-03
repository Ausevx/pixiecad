import { useCallback, useEffect, useMemo, useSyncExternalStore } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   A router for three routes and one query parameter.

   This exists instead of react-router because that is ~25 KB gzip and an
   unfixable advisory for a feature this app does not use (RSC actions), to
   solve a problem that is genuinely this small. The whole surface is: which
   of three screens am I on, which job, and is the compute drawer open.
   ───────────────────────────────────────────────────────────────────────── */

export type Route =
  | { kind: "jobs" }
  | { kind: "new" }
  | { kind: "job"; id: string }
  /** Anything we do not recognise, so the shell can render a real 404 rather
   *  than silently pretending the user asked for the jobs list. */
  | { kind: "unknown"; path: string };

/** The SPA is mounted under /ui/ so the legacy dashboard can keep serving /
 *  while this is built. Every path below is relative to that. */
const BASE = "/ui";

function stripBase(pathname: string): string {
  const p = pathname.startsWith(BASE) ? pathname.slice(BASE.length) : pathname;
  return p === "" ? "/" : p;
}

export function parseRoute(pathname: string): Route {
  const path = stripBase(pathname).replace(/\/+$/, "") || "/";
  if (path === "/") return { kind: "jobs" };
  if (path === "/new") return { kind: "new" };
  const job = /^\/jobs\/([A-Za-z0-9_-]+)$/.exec(path);
  if (job?.[1]) return { kind: "job", id: job[1] };
  return { kind: "unknown", path };
}

export const href = (route: Route): string => {
  switch (route.kind) {
    case "jobs":
      return `${BASE}/`;
    case "new":
      return `${BASE}/new`;
    case "job":
      return `${BASE}/jobs/${route.id}`;
    case "unknown":
      return `${BASE}${route.path}`;
  }
};

/* ── Subscription ─────────────────────────────────────────────────────────
   One store for both popstate (back/forward) and our own pushes, so a
   navigation from either source updates every subscriber identically. */

const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  window.addEventListener("popstate", onChange);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("popstate", onChange);
  };
}

// useSyncExternalStore compares snapshots by identity, so this must be a
// primitive. Returning a fresh {pathname, search} object each call would
// re-render on every store read, forever.
const snapshot = () => window.location.pathname + window.location.search;

export function useLocation(): { pathname: string; search: string } {
  const url = useSyncExternalStore(subscribe, snapshot, () => `${BASE}/`);
  return useMemo(() => {
    const q = url.indexOf("?");
    return q === -1
      ? { pathname: url, search: "" }
      : { pathname: url.slice(0, q), search: url.slice(q) };
  }, [url]);
}

export function useRoute(): Route {
  const { pathname } = useLocation();
  return useMemo(() => parseRoute(pathname), [pathname]);
}

export function navigate(to: string, options?: { replace?: boolean }): void {
  if (to === snapshot()) return;
  window.history[options?.replace ? "replaceState" : "pushState"]({}, "", to);
  notify();
}

export const go = (route: Route, options?: { replace?: boolean }) =>
  navigate(href(route) + window.location.search, options);

/* ── The compute drawer ───────────────────────────────────────────────────
   Drawer state lives in the query string rather than in React state so it is
   deep-linkable, survives a reload, and closes on the browser back button —
   all three of which a plain useState drawer gets wrong. */

export function useDrawer(): {
  open: boolean;
  setOpen: (open: boolean) => void;
  toggle: () => void;
} {
  const { pathname, search } = useLocation();
  const open = new URLSearchParams(search).get("compute") === "1";

  const setOpen = useCallback(
    (next: boolean) => {
      const params = new URLSearchParams(window.location.search);
      if (next) params.set("compute", "1");
      else params.delete("compute");
      const q = params.toString();
      // Opening pushes so Back closes the drawer; closing replaces so the
      // user is not forced to press Back twice to leave the page.
      navigate(window.location.pathname + (q ? `?${q}` : ""), { replace: !next });
    },
    // pathname participates so the callback is not stale after a navigation.
    [pathname],
  );

  return {
    open,
    setOpen,
    toggle: useCallback(() => setOpen(!open), [open, setOpen]),
  };
}

/* ── Link behaviour ───────────────────────────────────────────────────────
   Intercept in-app anchor clicks so navigation is client-side, while leaving
   every escape hatch a real anchor gives the user intact: modifier-clicks,
   middle-click, target=_blank, and download links must all behave natively.
   Download links especially — they are the entire point of this application. */

export function useLinkInterception(): void {
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0) return;
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

      const anchor = (e.target as Element | null)?.closest?.("a");
      if (!anchor) return;
      if (anchor.target && anchor.target !== "_self") return;
      if (anchor.hasAttribute("download")) return;
      if (anchor.getAttribute("rel")?.includes("external")) return;

      const url = new URL(anchor.href, window.location.href);
      if (url.origin !== window.location.origin) return;
      // API paths are real server resources (downloads, models); never hijack.
      if (!url.pathname.startsWith(BASE)) return;

      e.preventDefault();
      navigate(url.pathname + url.search);
    };

    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);
}
