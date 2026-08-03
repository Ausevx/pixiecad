import type React from "react";
import { useState, useRef, useEffect, useLayoutEffect, useId, useCallback } from 'react';
import { motion } from 'framer-motion';
import { signatures } from '@/lib/motion';
import { useReducedMotion } from '@/lib/hooks';
import { formatCount } from '@/lib/api';

/**
 * Severity styling rules based on line contents.
 */
function getLineClass(line: string): string {
  if (line.startsWith('WARNING:') || line.includes('[FAILED]')) {
    return 'text-fail';
  }
  if (line.includes('[SKIPPED]')) {
    return 'text-ink-faint';
  }
  if (line.includes('[OK]')) {
    return 'text-ok';
  }
  return 'text-ink-dim';
}

export function LogPanel(props: {
  lines: string[];
  /** True while the job is still producing output. */
  live: boolean;
  /** Collapsed initially; the component owns the state after that. */
  defaultOpen?: boolean;
  className?: string;
}): React.JSX.Element {
  const { lines, live, defaultOpen = false, className = '' } = props;

  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  const scrollRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);
  const prevLinesLengthRef = useRef(lines.length);
  const copiedTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const logRegionId = useId();
  const isReducedMotion = useReducedMotion();

  // Scroll to bottom helper
  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
      isAtBottomRef.current = true;
      setUnreadCount(0);
    }
  }, []);

  // Track scroll position to update tail-follow status (40px threshold)
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;

    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distanceFromBottom <= 40;
    isAtBottomRef.current = atBottom;

    if (atBottom) {
      setUnreadCount(0);
    }
  }, []);

  // Maintain scroll position or tail-follow on lines update
  useLayoutEffect(() => {
    if (!isOpen || !scrollRef.current) return;

    const currentLength = lines.length;
    const prevLength = prevLinesLengthRef.current;
    const diff = currentLength - prevLength;
    prevLinesLengthRef.current = currentLength;

    if (diff > 0) {
      if (isAtBottomRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        setUnreadCount(0);
      } else {
        setUnreadCount((prev) => prev + diff);
      }
    } else if (currentLength < prevLength) {
      setUnreadCount(0);
    }
  }, [lines, isOpen]);

  // Ensure view scrolls to bottom when first expanded if tail-following was active
  useLayoutEffect(() => {
    if (isOpen && scrollRef.current && isAtBottomRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [isOpen]);

  // Clipboard copy handler with temporary feedback
  const handleCopy = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      const content = lines.join('\n');
      navigator.clipboard.writeText(content).catch(() => {
        // Fallback for restricted clipboard contexts
      });
      setCopied(true);
      if (copiedTimeoutRef.current) {
        clearTimeout(copiedTimeoutRef.current);
      }
      copiedTimeoutRef.current = setTimeout(() => {
        setCopied(false);
      }, 2000);
    },
    [lines]
  );

  useEffect(() => {
    return () => {
      if (copiedTimeoutRef.current) {
        clearTimeout(copiedTimeoutRef.current);
      }
    };
  }, []);

  const formattedCount = formatCount(lines.length);
  const lineLabel = lines.length === 1 ? 'line' : 'lines';

  return (
    <div
      className={`font-mono text-xs text-ink-dim select-none flex flex-col ${className}`}
    >
      {/* Top Frame Rule & Header Bar */}
      <div className="flex flex-col bg-panel rounded-sharp overflow-hidden">
        {/* Frame Top Rule carrying box-drawing characters and title */}
        <div
          className="flex items-center text-ink-faint font-mono text-xs select-none px-1 pt-1"
          aria-hidden="true"
        >
          <span>┌──&nbsp;</span>
          <span className="text-ink font-semibold tracking-wider">LOG</span>
          <span>&nbsp;</span>
          <span className="flex-1 overflow-hidden whitespace-nowrap opacity-30">
            ────────────────────────────────────────────────────────────────────────────────────────────────────
          </span>
          <span>&nbsp;</span>
          <span className="text-ink-dim tabular-nums font-mono">
            {formattedCount} {lineLabel}
          </span>
          <span>&nbsp;</span>
          <span className="w-12 overflow-hidden whitespace-nowrap opacity-30">
            ──────────
          </span>
          <span>┐</span>
        </div>

        {/* Header Controls & Toggle Button */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-panel border-x border-transparent">
          {/* Accessible Toggle Button */}
          <button
            type="button"
            onClick={() => setIsOpen((prev) => !prev)}
            aria-expanded={isOpen}
            aria-controls={logRegionId}
            className="flex items-center gap-2 text-ink hover:text-accent transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent rounded-sharp text-xs font-mono font-medium cursor-pointer"
          >
            <span aria-hidden="true" className="text-accent text-xs">
              {isOpen ? '▼' : '►'}
            </span>
            <span>{isOpen ? 'Collapse Logs' : 'Expand Logs'}</span>

            {/* Live Output Pulse Dot */}
            {live && (
              <span className="flex items-center gap-1.5 text-accent text-[11px] ml-2 font-mono">
                {isReducedMotion ? (
                  <span className="w-2 h-2 rounded-full bg-accent inline-block" />
                ) : (
                  <motion.span
                    animate={signatures.running}
                    className="w-2 h-2 rounded-full bg-accent inline-block"
                  />
                )}
                <span className="uppercase tracking-wider text-[10px]">LIVE</span>
              </span>
            )}
          </button>

          {/* Accessible Copy-All Button */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopy}
              aria-label="Copy all log output to clipboard"
              className="px-2 py-0.5 text-[11px] font-mono text-ink-dim hover:text-ink hover:bg-hover active:bg-raised border border-rule hover:border-rule-bright rounded-sharp transition-colors flex items-center gap-1.5 cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
            >
              {copied ? (
                <span className="text-ok font-semibold">COPIED</span>
              ) : (
                <span>COPY ALL</span>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Expanded Content Section */}
      {isOpen && (
        <>
          {/* Middle Frame Separator Rule */}
          <div
            className="flex items-center text-ink-faint font-mono text-xs select-none px-1"
            aria-hidden="true"
          >
            <span>├</span>
            <span className="flex-1 overflow-hidden whitespace-nowrap opacity-30">
              ────────────────────────────────────────────────────────────────────────────────────────────────────
            </span>
            <span>┤</span>
          </div>

          {/* Log Scroll Region with Side Borders */}
          <div className="relative flex items-stretch bg-void">
            {/* Decorative Left Border Character */}
            <div
              className="select-none text-ink-faint font-mono text-xs px-1 py-1 flex flex-col justify-between overflow-hidden opacity-30"
              aria-hidden="true"
            >
              <span>│</span>
            </div>

            {/* Scroll Container (Plain elements only for performance) */}
            <div
              id={logRegionId}
              ref={scrollRef}
              onScroll={handleScroll}
              role="log"
              tabIndex={0}
              className="flex-1 max-h-80 overflow-y-auto py-1 px-2 font-mono text-xs leading-relaxed focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
            >
              {lines.length === 0 ? (
                <div className="text-ink-faint italic py-2">No output recorded.</div>
              ) : (
                lines.map((line, index) => {
                  const styleClass = getLineClass(line);
                  const isIndented = line.startsWith('  ');
                  return (
                    <div
                      key={index}
                      className={`whitespace-pre-wrap break-words ${styleClass} ${
                        isIndented ? 'pl-4' : ''
                      }`}
                    >
                      {line}
                    </div>
                  );
                })
              )}
            </div>

            {/* Decorative Right Border Character */}
            <div
              className="select-none text-ink-faint font-mono text-xs px-1 py-1 flex flex-col justify-between overflow-hidden opacity-30"
              aria-hidden="true"
            >
              <span>│</span>
            </div>

            {/* Unobtrusive New Lines Jump Affordance */}
            {unreadCount > 0 && (
              <div className="absolute bottom-3 right-4 z-10">
                <button
                  type="button"
                  onClick={scrollToBottom}
                  aria-label={`Jump to bottom, ${unreadCount} new ${
                    unreadCount === 1 ? 'line' : 'lines'
                  }`}
                  className="bg-raised hover:bg-hover active:bg-panel text-accent border border-accent-dim rounded-sharp px-2.5 py-1 text-xs font-mono shadow-md flex items-center gap-1.5 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
                >
                  <span>↓</span>
                  <span>
                    {formatCount(unreadCount)} new
                  </span>
                </button>
              </div>
            )}
          </div>
        </>
      )}

      {/* Frame Bottom Rule */}
      <div
        className="flex items-center text-ink-faint font-mono text-xs select-none px-1 pb-1"
        aria-hidden="true"
      >
        <span>└</span>
        <span className="flex-1 overflow-hidden whitespace-nowrap opacity-30">
          ────────────────────────────────────────────────────────────────────────────────────────────────────
        </span>
        <span>┘</span>
      </div>
    </div>
  );
}
