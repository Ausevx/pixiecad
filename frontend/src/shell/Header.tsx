import { motion } from "motion/react";
import { Logo } from "./Logo";
import { ThemeToggle } from "./ThemeToggle";
import { VmPill } from "./VmPill";
import { href, useRoute, type Route } from "@/lib/router";
import { snap } from "@/lib/motion";
import { useReducedMotion } from "@/lib/hooks";

const NAV: { route: Route; label: string; key: string }[] = [
  { route: { kind: "jobs" }, label: "jobs", key: "jobs" },
  { route: { kind: "new" }, label: "new", key: "new" },
];

/** The active tab's underline is a shared layout element, so moving between
 *  tabs slides one rule rather than cross-fading two. */
function NavLink({ route, label, active }: { route: Route; label: string; active: boolean }) {
  return (
    <a
      href={href(route)}
      aria-current={active ? "page" : undefined}
      className={`relative px-2 py-1 font-mono text-[12px] uppercase tracking-widest transition-colors ${
        active ? "text-ink" : "text-ink-faint hover:text-ink-dim"
      }`}
    >
      {label}
      {active && (
        <motion.span
          layoutId="nav-underline"
          transition={snap}
          aria-hidden="true"
          className="absolute inset-x-1 -bottom-px h-px bg-accent"
        />
      )}
    </a>
  );
}

export function Header({ onOpenCompute }: { onOpenCompute: () => void }) {
  const route = useRoute();
  const reduced = useReducedMotion();
  // The job detail route is reached from the jobs list, so it keeps that tab
  // lit rather than leaving no tab active and the user feeling unmoored.
  const activeKey = route.kind === "new" ? "new" : "jobs";

  return (
    <header className="sticky top-0 z-30 border-b border-rule bg-void/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-6 px-5">
        <a
          href={href({ kind: "jobs" })}
          className="shrink-0"
          aria-label="PixieCAD, go to jobs"
        >
          <Logo />
        </a>

        <nav aria-label="Main" className="flex items-center gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.key}
              route={item.route}
              label={item.label}
              active={activeKey === item.key}
            />
          ))}
        </nav>

        <div className="flex-1" />

        <motion.a
          href={href({ kind: "new" })}
          whileHover={reduced ? undefined : { scale: 1.03 }}
          whileTap={reduced ? undefined : { scale: 0.97 }}
          transition={snap}
          className="rounded-panel border border-accent-dim bg-accent-wash px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-accent hover:bg-accent hover:text-void"
        >
          + new job
        </motion.a>

        <ThemeToggle />
        <VmPill onOpen={onOpenCompute} />
      </div>
    </header>
  );
}
