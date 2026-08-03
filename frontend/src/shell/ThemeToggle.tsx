import { motion } from "motion/react";
import { snap } from "@/lib/motion";
import { useReducedMotion } from "@/lib/hooks";
import { useTheme, type ThemeChoice } from "@/lib/theme";

const GLYPH: Record<ThemeChoice, string> = {
  system: "◐",
  dark: "●",
  light: "○",
};

const LABEL: Record<ThemeChoice, string> = {
  system: "system",
  dark: "dark",
  light: "light",
};

/** One button cycling system → dark → light. A three-way segmented control
 *  would be more discoverable, but this is a setting people touch once; the
 *  header is worth more to the work than to a preference. The current state
 *  is always announced, so nothing is hidden from a screen reader. */
export function ThemeToggle() {
  const { choice, cycle } = useTheme();
  const reduced = useReducedMotion();

  const next: ThemeChoice =
    choice === "system" ? "dark" : choice === "dark" ? "light" : "system";

  return (
    <motion.button
      type="button"
      onClick={cycle}
      whileHover={reduced ? undefined : { scale: 1.05 }}
      whileTap={reduced ? undefined : { scale: 0.95 }}
      transition={snap}
      aria-label={`Theme: ${LABEL[choice]}. Switch to ${LABEL[next]}.`}
      title={`Theme: ${LABEL[choice]}`}
      className="rounded-panel border border-rule bg-raised px-2 py-1 font-mono text-[11px] text-ink-dim hover:border-accent-dim hover:text-accent"
    >
      <span aria-hidden="true">{GLYPH[choice]}</span>
      <span className="sr-only">{LABEL[choice]}</span>
    </motion.button>
  );
}

/** Exposed so the theme can also be cycled from a keyboard shortcut without
 *  the toggle having to be mounted. */
export { useTheme, setTheme } from "@/lib/theme";
