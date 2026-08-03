import type React from "react";
import { useState, useEffect, useRef, useId, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { ViewTag } from "@/lib/types";
import { imageToAscii } from "@/lib/ascii";
import { formatBytes } from "@/lib/api";
import { useReducedMotion } from "@/lib/hooks";
import { snap, drop, listVariants, listItemVariants } from "@/lib/motion";

export interface PhotoEntry {
  id: string;
  file: File;
  /** front | back | left | right, or undefined for untagged. */
  tag?: ViewTag;
}

// Module-level cache for ASCII results keyed by photo entry ID
const asciiCache = new Map<
  string,
  { rows: string[]; cols: number; rowCount: number }
>();

const VIEW_TAG_OPTIONS: { value: ViewTag; label: string }[] = [
  { value: "front", label: "front" },
  { value: "back", label: "back" },
  { value: "left", label: "left" },
  { value: "right", label: "right" },
];

function PhotoCard({
  photo,
  allPhotos,
  onTagChange,
  onRemove,
}: {
  photo: PhotoEntry;
  allPhotos: PhotoEntry[];
  onTagChange: (id: string, tag?: ViewTag) => void;
  onRemove: (id: string) => void;
}): React.JSX.Element {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [ascii, setAscii] = useState<{
    rows: string[];
    cols: number;
    rowCount: number;
  } | null>(() => asciiCache.get(photo.id) || null);
  const [isLoadingAscii, setIsLoadingAscii] = useState<boolean>(
    !asciiCache.has(photo.id)
  );
  const isReducedMotion = useReducedMotion();
  const selectId = useId();

  // Create and revoke object URL for the real image preview on removal/unmount
  useEffect(() => {
    const url = URL.createObjectURL(photo.file);
    setImageUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [photo.file]);

  // Convert image to ASCII via Web Worker, handling cancellation and caching
  useEffect(() => {
    if (asciiCache.has(photo.id)) {
      setAscii(asciiCache.get(photo.id)!);
      setIsLoadingAscii(false);
      return;
    }

    const controller = new AbortController();
    let isMounted = true;

    async function runConversion() {
      try {
        const result = await imageToAscii(photo.file, {
          cols: 40,
          ramp: "ascii",
          signal: controller.signal,
        });

        if (isMounted && result) {
          asciiCache.set(photo.id, result);
          setAscii(result);
          setIsLoadingAscii(false);
        }
      } catch (error: unknown) {
        // Swallowing AbortError when conversion is cancelled (e.g. photo removed)
        if (
          (error instanceof DOMException && error.name === "AbortError") ||
          controller.signal.aborted
        ) {
          return;
        }
        if (isMounted) {
          setIsLoadingAscii(false);
        }
      }
    }

    runConversion();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [photo.id, photo.file]);

  const isTagDisabled = useCallback(
    (tagValue: ViewTag) => {
      return allPhotos.some((p) => p.id !== photo.id && p.tag === tagValue);
    },
    [allPhotos, photo.id]
  );

  return (
    <motion.li
      variants={listItemVariants}
      transition={isReducedMotion ? undefined : drop}
      className="group relative flex flex-col gap-2 rounded-panel border border-rule bg-panel p-3 transition-colors hover:border-rule-bright focus-within:border-rule-bright"
    >
      {/* ASCII / Image Preview Box */}
      <div className="relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-sharp bg-void p-1 select-none">
        {/* Hover/Focus image overlay */}
        {imageUrl && (
          <img
            src={imageUrl}
            alt=""
            className="absolute inset-0 z-10 h-full w-full object-contain opacity-0 transition-opacity duration-180 group-hover:opacity-100 group-focus-within:opacity-100 pointer-events-none"
          />
        )}

        {/* ASCII Preview */}
        {ascii ? (
          <pre
            aria-hidden="true"
            className="font-mono text-[5px] leading-[1.05] text-ink-dim select-none overflow-hidden text-center whitespace-pre pointer-events-none"
          >
            {ascii.rows.join("\n")}
          </pre>
        ) : isLoadingAscii ? (
          <div
            className={`h-full w-full rounded-sharp bg-raised border border-rule ${
              isReducedMotion ? "" : "animate-pulse"
            }`}
            aria-hidden="true"
          />
        ) : (
          <div className="font-mono text-xs text-ink-faint">
            [preview unavailable]
          </div>
        )}

        {/* Screen-reader accessible alternative */}
        <span className="sr-only">
          Photo preview for {photo.file.name}
          {photo.tag ? `, view tag: ${photo.tag}` : ""}
        </span>
      </div>

      {/* File Info, Tag Selector & Remove Action */}
      <div className="flex flex-col gap-1.5 text-xs">
        <div className="flex items-center justify-between gap-2">
          <span
            className="truncate font-mono font-medium text-ink"
            title={photo.file.name}
          >
            {photo.file.name}
          </span>
          <span className="shrink-0 font-mono text-ink-faint">
            {formatBytes(photo.file.size)}
          </span>
        </div>

        <div className="flex items-center justify-between gap-2 pt-1 border-t border-rule">
          <div className="flex items-center gap-1.5">
            <label
              htmlFor={selectId}
              className="font-mono text-[10px] uppercase tracking-wider text-ink-faint"
            >
              Tag:
            </label>
            <select
              id={selectId}
              value={photo.tag ?? ""}
              onChange={(e) => {
                const val = e.target.value;
                onTagChange(
                  photo.id,
                  val === "" ? undefined : (val as ViewTag)
                );
              }}
              className="rounded-sharp border border-rule bg-raised px-2 py-0.5 font-mono text-xs text-ink focus:border-accent focus:outline-none"
            >
              <option value="">—</option>
              {VIEW_TAG_OPTIONS.map((opt) => (
                <option
                  key={opt.value}
                  value={opt.value}
                  disabled={isTagDisabled(opt.value)}
                >
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            onClick={() => onRemove(photo.id)}
            aria-label={`Remove ${photo.file.name}`}
            className="rounded-sharp px-2 py-0.5 font-mono text-xs font-semibold text-ink-dim hover:bg-hover hover:text-fail focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-fail transition-colors"
          >
            × REMOVE
          </button>
        </div>
      </div>
    </motion.li>
  );
}

export function PhotoDrop({
  photos,
  onChange,
  className = "",
}: {
  photos: PhotoEntry[];
  onChange: (next: PhotoEntry[]) => void;
  className?: string;
}): React.JSX.Element {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isReducedMotion = useReducedMotion();

  const totalSize = photos.reduce((acc, p) => acc + p.file.size, 0);

  const processFiles = useCallback(
    (files: FileList | File[]) => {
      const imageFiles = Array.from(files).filter((file) =>
        file.type.startsWith("image/")
      );
      if (imageFiles.length === 0) return;

      const newEntries: PhotoEntry[] = imageFiles.map((file) => ({
        id: `${file.name}-${file.size}-${Date.now()}-${Math.random()
          .toString(36)
          .substring(2, 9)}`,
        file,
        tag: undefined,
      }));

      onChange([...photos, ...newEntries]);
    },
    [photos, onChange]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const handleTagChange = (id: string, newTag?: ViewTag) => {
    if (newTag && photos.some((p) => p.id !== id && p.tag === newTag)) {
      return;
    }
    const nextPhotos = photos.map((p) =>
      p.id === id ? { ...p, tag: newTag } : p
    );
    onChange(nextPhotos);
  };

  const handleRemove = (id: string) => {
    const nextPhotos = photos.filter((p) => p.id !== id);
    onChange(nextPhotos);
  };

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="sr-only"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            processFiles(e.target.files);
            e.target.value = "";
          }
        }}
      />

      {/* Dropzone Button */}
      <motion.button
        type="button"
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        animate={{
          scale: isDragOver && !isReducedMotion ? 0.99 : 1,
        }}
        transition={isReducedMotion ? undefined : snap}
        className={`group relative flex min-h-[140px] w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-panel border-2 border-dashed p-6 text-center transition-colors focus-visible:outline-none ${
          isDragOver
            ? "border-accent-dim bg-accent-wash"
            : "border-rule bg-panel hover:border-rule-bright hover:bg-raised"
        }`}
      >
        <div className="flex flex-col items-center gap-1">
          <span className="font-mono text-sm font-semibold text-ink group-hover:text-accent">
            DROP PHOTOS HERE OR CLICK TO BROWSE
          </span>
          <span className="font-mono text-xs text-ink-dim">
            Accepts image files (JPEG, PNG, WEBP, etc.)
          </span>
        </div>

        {/* Monospace photo count & total memory summary */}
        <div className="mt-1 font-mono text-xs text-ink-faint">
          {photos.length} photo{photos.length === 1 ? "" : "s"}
          {photos.length > 0 ? ` · ${formatBytes(totalSize)}` : ""}
        </div>
      </motion.button>

      {/* Photo List Grid */}
      {photos.length > 0 && (
        <motion.ul
          variants={listVariants}
          initial="initial"
          animate="animate"
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
        >
          <AnimatePresence mode="popLayout">
            {photos.map((photo) => (
              <PhotoCard
                key={photo.id}
                photo={photo}
                allPhotos={photos}
                onTagChange={handleTagChange}
                onRemove={handleRemove}
              />
            ))}
          </AnimatePresence>
        </motion.ul>
      )}
    </div>
  );
}
