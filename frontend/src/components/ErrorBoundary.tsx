import { Component, type ErrorInfo, type ReactNode } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   A boundary so one broken panel cannot take the application with it.

   This exists because it already happened: three.js throws when a WebGL
   context cannot be created — a blocklisted driver, a remote session, a
   browser with it turned off — and with nothing to catch it, React unmounted
   the whole tree. The result was a completely blank page for a finished job
   whose STL was sitting on disk one click away.

   The rule this encodes: a decorative or secondary panel failing must never
   cost the user access to their artifacts.
   ───────────────────────────────────────────────────────────────────────── */

interface Props {
  children: ReactNode;
  /** Shown in place of the children. Receives the error for context. */
  fallback?: (error: Error) => ReactNode;
  /** Changing this resets the boundary — used to retry on navigation rather
   *  than leaving a stale error on screen after the user moves on. */
  resetKey?: string;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidUpdate(prev: Props): void {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    // The console is the only reporting channel a local-only tool has, and
    // swallowing this silently would make the fallback impossible to debug.
    console.error("[pixiecad] component crashed:", error, info.componentStack);
  }

  override render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    if (this.props.fallback) return this.props.fallback(error);

    return (
      <div className="rounded-panel border border-[#5c1d1d] bg-[#2a0d0d] p-4">
        <p className="font-mono text-[11px] uppercase tracking-widest text-fail">
          this panel failed
        </p>
        <p className="mt-1 font-mono text-[11px] leading-relaxed break-words text-ink-dim">
          {error.message}
        </p>
      </div>
    );
  }
}
