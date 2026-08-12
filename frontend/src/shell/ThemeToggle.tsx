import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { snap } from "@/lib/motion";
import { useReducedMotion } from "@/lib/hooks";
import { useTheme, THEMES, themeMeta, type ThemeChoice } from "@/lib/theme";

/** "system" leads because it is the default and the only choice that keeps
 *  tracking the OS; the named themes follow in registry order. */
const CHOICES: ThemeChoice[] = ["system", ...THEMES.map((t) => t.id)];

/** A menu rather than the cycling button this replaced. One button was right
 *  for three states and becomes unusable at seven — reaching "sepia" would mean
 *  clicking through four themes you did not want, repainting the page each
 *  time. The current state is always announced, so nothing is hidden from a
 *  screen reader. */
export function ThemeToggle() {
  const { choice, setTheme } = useTheme();
  const reduced = useReducedMotion();
  const [open, setOpen] = useState(false);

  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (e: PointerEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [open]);

  useEffect(() => {
    if (open) {
      const idx = CHOICES.indexOf(choice);
      requestAnimationFrame(() => {
        itemRefs.current[idx]?.focus();
      });
    }
  }, [open, choice]);

  const handleTriggerKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setOpen(true);
    }
  };

  const handleItemKeyDown = (e: React.KeyboardEvent, index: number) => {
    switch (e.key) {
      case "Escape":
        e.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
        break;
      case "ArrowDown":
        e.preventDefault();
        itemRefs.current[(index + 1) % CHOICES.length]?.focus();
        break;
      case "ArrowUp":
        e.preventDefault();
        itemRefs.current[(index - 1 + CHOICES.length) % CHOICES.length]?.focus();
        break;
      case "Home":
        e.preventDefault();
        itemRefs.current[0]?.focus();
        break;
      case "End":
        e.preventDefault();
        itemRefs.current[CHOICES.length - 1]?.focus();
        break;
      case "Enter":
      case " ": {
        e.preventDefault();
        const choiceToSet = CHOICES[index];
        if (choiceToSet) setTheme(choiceToSet);
        setOpen(false);
        triggerRef.current?.focus();
        break;
      }
    }
  };

  const getLabel = (c: ThemeChoice) => (c === "system" ? "system" : themeMeta(c).label);

  return (
    <div className="relative">
      <motion.button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        onKeyDown={handleTriggerKeyDown}
        whileHover={reduced ? undefined : { scale: 1.05 }}
        whileTap={reduced ? undefined : { scale: 0.95 }}
        transition={snap}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Theme: ${getLabel(choice)}. Change theme.`}
        title={`Theme: ${getLabel(choice)}`}
        className="rounded-panel border border-rule bg-raised px-2 py-1 font-mono text-[11px] text-ink-dim hover:border-accent-dim hover:text-accent flex items-center justify-center"
      >
        {choice === "system" ? (
          <span aria-hidden="true" className="leading-none">◐</span>
        ) : (
          <span
            aria-hidden="true"
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ background: themeMeta(choice).swatch }}
          />
        )}
        <span className="sr-only">{getLabel(choice)}</span>
      </motion.button>
      <AnimatePresence>
        {open && (
          <motion.div
            ref={menuRef}
            role="menu"
            initial={reduced ? undefined : { opacity: 0, scale: 0.95 }}
            animate={reduced ? undefined : { opacity: 1, scale: 1 }}
            exit={reduced ? undefined : { opacity: 0, scale: 0.95 }}
            transition={snap}
            /* A hairline edge, not a drop shadow: on the near-black grounds a
               box-shadow reads as mud, which is why the rest of the design
               separates surfaces with a rule. z-40 clears the sticky header. */
            className="absolute top-full mt-1 right-0 z-40 flex min-w-[140px] flex-col rounded-panel border border-rule bg-panel py-1"
          >
            {CHOICES.map((c, i) => {
              const isSelected = c === choice;
              return (
                <button
                  key={c}
                  ref={(el) => {
                    itemRefs.current[i] = el;
                  }}
                  role="menuitemradio"
                  aria-checked={isSelected}
                  onClick={() => {
                    setTheme(c);
                    setOpen(false);
                    triggerRef.current?.focus();
                  }}
                  onKeyDown={(e) => handleItemKeyDown(e, i)}
                  className="px-3 py-1.5 flex items-center justify-between text-left font-mono text-[11px] text-ink-dim hover:bg-hover focus:bg-hover hover:text-ink focus:text-ink focus:outline-none"
                  tabIndex={-1}
                >
                  <div className="flex items-center gap-2">
                    {c === "system" ? (
                      <span aria-hidden="true" className="leading-none">◐</span>
                    ) : (
                      <span
                        aria-hidden="true"
                        className="inline-block w-2.5 h-2.5 rounded-full"
                        style={{ background: themeMeta(c).swatch }}
                      />
                    )}
                    <span>{getLabel(c)}</span>
                  </div>
                  {isSelected && <span className="text-accent leading-none" aria-hidden="true">✓</span>}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Exposed so the theme can also be set from a keyboard shortcut without the
 *  picker having to be mounted. */
export { useTheme, setTheme } from "@/lib/theme";
