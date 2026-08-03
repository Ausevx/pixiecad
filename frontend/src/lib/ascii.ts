import AsciiWorker from "./ascii.worker.ts?worker";

export interface AsciiResult {
  rows: string[]; // one string per character row
  cols: number;
  rowCount: number;
}

export interface AsciiOptions {
  cols?: number; // default 72
  ramp?: "ascii" | "blocks"; // default "ascii"
  signal?: AbortSignal; // cancel an in-flight conversion
}

export interface WorkerRequest {
  id: number;
  bitmap: ImageBitmap;
  cols: number;
  ramp: "ascii" | "blocks";
}

export type WorkerResponse =
  | { id: number; success: true; result: AsciiResult }
  | { id: number; success: false; error: string };

interface PendingRequest {
  resolve: (result: AsciiResult) => void;
  reject: (reason: unknown) => void;
  signal?: AbortSignal;
  onAbort?: () => void;
}

let workerInstance: Worker | null = null;
let nextRequestId = 1;
const pendingRequests = new Map<number, PendingRequest>();

function handleWorkerMessage(event: MessageEvent<WorkerResponse>): void {
  const data = event.data;
  if (!data || typeof data.id !== "number") {
    return;
  }

  const pending = pendingRequests.get(data.id);
  if (!pending) {
    // Request was aborted or cancelled; reply is safely discarded.
    return;
  }

  pendingRequests.delete(data.id);

  if (data.success) {
    pending.resolve(data.result);
  } else {
    pending.reject(new Error(data.error));
  }
}

function handleWorkerError(event: ErrorEvent): void {
  const err = new Error(event.message || "ASCII worker encountered an uncaught error.");
  for (const pending of pendingRequests.values()) {
    pending.reject(err);
  }
  pendingRequests.clear();
}

function getWorker(): Worker {
  if (!workerInstance) {
    workerInstance = new AsciiWorker();
    workerInstance.onmessage = handleWorkerMessage;
    workerInstance.onerror = handleWorkerError;
  }
  return workerInstance;
}

/** Convert an image file to ASCII. Resolves once the worker replies. */
export async function imageToAscii(
  source: File | Blob,
  options?: AsciiOptions,
): Promise<AsciiResult> {
  const signal = options?.signal;
  if (signal?.aborted) {
    throw new DOMException("The operation was aborted.", "AbortError");
  }

  const cols = options?.cols ?? 72;
  const ramp = options?.ramp ?? "ascii";

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(source);
  } catch (err) {
    throw new Error(
      `Failed to create ImageBitmap: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (signal?.aborted) {
    bitmap.close();
    throw new DOMException("The operation was aborted.", "AbortError");
  }

  const id = nextRequestId++;
  const worker = getWorker();

  return new Promise<AsciiResult>((resolve, reject) => {
    let onAbort: (() => void) | undefined;

    if (signal) {
      onAbort = () => {
        pendingRequests.delete(id);
        reject(new DOMException("The operation was aborted.", "AbortError"));
      };
      signal.addEventListener("abort", onAbort, { once: true });
    }

    pendingRequests.set(id, {
      resolve: (result) => {
        if (signal && onAbort) {
          signal.removeEventListener("abort", onAbort);
        }
        resolve(result);
      },
      reject: (err) => {
        if (signal && onAbort) {
          signal.removeEventListener("abort", onAbort);
        }
        reject(err);
      },
    });

    try {
      const request: WorkerRequest = { id, bitmap, cols, ramp };
      worker.postMessage(request, [bitmap]);
    } catch (err) {
      if (signal && onAbort) {
        signal.removeEventListener("abort", onAbort);
      }
      pendingRequests.delete(id);
      bitmap.close();
      reject(err);
    }
  });
}