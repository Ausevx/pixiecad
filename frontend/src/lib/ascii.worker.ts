import type { AsciiResult, WorkerRequest, WorkerResponse } from "./ascii";

// Pre-computed lookup table for sRGB channel linearisation (0-255 -> 0.0-1.0 linear).
// sRGB channels must be linearised prior to perceptual luminance calculation because
// sRGB is gamma-encoded (~2.2 gamma). Naive weighted sums directly on gamma values produce
// muddy midtones and incorrect contrast transitions.
const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

// Re-encodes linear luminance back to sRGB space for accurate human perception matching.
function linearToSrgb(lLin: number): number {
  if (lLin <= 0.0031308) {
    return 12.92 * lLin;
  }
  return 1.055 * Math.pow(lLin, 1 / 2.4) - 0.055;
}

// 4x4 Bayer matrix for ordered dithering. Dithering breaks up solid slabs of identical
// characters across continuous gradients, preserving structural details as texture.
const BAYER_4X4: readonly (readonly number[])[] = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
];

const RAMPS = {
  ascii: " .:-=+*#%@",
  blocks: " ░▒▓█",
} as const;

function processImage(
  bitmap: ImageBitmap,
  cols: number,
  rampType: "ascii" | "blocks",
): AsciiResult {
  const ramp = RAMPS[rampType];

  // Monospace font cells have an aspect ratio of approximately 1:2 (width is ~50% of height).
  // Without adjusting for cell aspect ratio, a square image would render twice as tall as it is wide.
  // We multiply row count by 0.5 (or divide height by width ratio) so visual aspect ratio is preserved.
  const cellAspect = 0.5;
  const imageAspect = bitmap.height / bitmap.width;
  const rowCount = Math.max(1, Math.round(cols * imageAspect * cellAspect));

  const canvas = new OffscreenCanvas(cols, rowCount);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to obtain 2D context from OffscreenCanvas.");
  }

  ctx.drawImage(bitmap, 0, 0, cols, rowCount);
  const imgData = ctx.getImageData(0, 0, cols, rowCount);
  const data = imgData.data;

  const totalPixels = cols * rowCount;
  const rawLuminance = new Float32Array(totalPixels);

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4] ?? 0;
    const g = data[i * 4 + 1] ?? 0;
    const b = data[i * 4 + 2] ?? 0;

    const rLin = SRGB_TO_LINEAR[r] ?? 0;
    const gLin = SRGB_TO_LINEAR[g] ?? 0;
    const bLin = SRGB_TO_LINEAR[b] ?? 0;

    // ITU-R BT.709 coefficients for linear RGB perceptual luminance calculation
    const lLin = 0.2126 * rLin + 0.7152 * gLin + 0.0722 * bLin;
    rawLuminance[i] = Math.max(0, Math.min(1, linearToSrgb(lLin)));
  }

  // Contrast normalisation: sample 2nd and 98th percentiles to stretch dynamic range.
  // Photos of CAD models or objects on white/neutral backgrounds often use only a slice
  // of the luminance spectrum; percentile stretching ensures full use of the ASCII ramp.
  const sortedLum = new Float32Array(rawLuminance);
  sortedLum.sort();

  const p2Index = Math.floor(totalPixels * 0.02);
  const p98Index = Math.min(totalPixels - 1, Math.floor(totalPixels * 0.98));

  const p2 = sortedLum[p2Index] ?? 0;
  const p98 = sortedLum[p98Index] ?? 1;
  const range = p98 > p2 ? p98 - p2 : 1;

  const rows: string[] = new Array(rowCount);
  const ditherScale = 1.0 / ramp.length;

  for (let y = 0; y < rowCount; y++) {
    let rowStr = "";
    const bayerRow = BAYER_4X4[y % 4] ?? BAYER_4X4[0]!;

    for (let x = 0; x < cols; x++) {
      const idx = y * cols + x;
      const lum = rawLuminance[idx] ?? 0;

      // Stretch contrast to [0, 1] range based on percentiles
      const normLum = Math.max(0, Math.min(1, (lum - p2) / range));

      // Apply ordered dithering
      const bayerVal = bayerRow[x % 4] ?? 0;
      const dither = ((bayerVal + 0.5) / 16 - 0.5) * ditherScale;
      const ditheredLum = Math.max(0, Math.min(0.9999, normLum + dither));

      const rampIdx = Math.floor(ditheredLum * ramp.length);
      rowStr += ramp.charAt(rampIdx);
    }
    rows[y] = rowStr;
  }

  return {
    rows,
    cols,
    rowCount,
  };
}

self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const { id, bitmap, cols, ramp } = event.data;

  try {
    const result = processImage(bitmap, cols, ramp);
    // Close transferred ImageBitmap immediately to release GPU/system memory
    bitmap.close();

    const response: WorkerResponse = { id, success: true, result };
    self.postMessage(response);
  } catch (err) {
    try {
      bitmap.close();
    } catch {
      // Ignore secondary close error
    }

    const response: WorkerResponse = {
      id,
      success: false,
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};