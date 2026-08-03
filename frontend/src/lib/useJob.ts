import { useCallback, useEffect, useRef, useState } from "react";
import { getJob, getJobFiles } from "./api";
import { usePoll } from "./hooks";
import { isTerminal, type JobDetail, type JobFiles } from "./types";

/* ─────────────────────────────────────────────────────────────────────────
   One job's live state.

   The shape of the work this displays is: seconds of upload, then many
   minutes of GPU work during which the user leaves and comes back. So this
   hook has to be resumable rather than continuous — it assumes nothing about
   having watched the whole run, and it holds a cursor into the log so that
   coming back after ten minutes costs one request, not a re-download of
   everything that happened while away.
   ───────────────────────────────────────────────────────────────────────── */

export interface JobState {
  job: JobDetail | null;
  log: string[];
  files: JobFiles | null;
  loading: boolean;
  /** 404 means the job genuinely does not exist — a deleted job, or a link
   *  from before the server learned to rehydrate. Distinguished from a
   *  transport error because the two need completely different screens. */
  notFound: boolean;
  error: string | null;
  refresh: () => void;
}

export function useJob(jobId: string): JobState {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [files, setFiles] = useState<JobFiles | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The accumulated log and the cursor into it. Kept in a ref so appending
  // does not depend on the previous render having committed.
  const logRef = useRef<string[]>([]);
  const [log, setLog] = useState<string[]>([]);
  // Files are re-listed only when the job actually finishes, not on every
  // poll: listing walks the output directory on disk.
  const filesFetchedFor = useRef<string | null>(null);
  const [nonce, setNonce] = useState(0);

  // Switching jobs must not show the previous job's log for even one frame.
  useEffect(() => {
    logRef.current = [];
    setLog([]);
    setJob(null);
    setFiles(null);
    setLoading(true);
    setNotFound(false);
    setError(null);
    filesFetchedFor.current = null;
  }, [jobId, nonce]);

  const poll = useCallback(
    async (signal: AbortSignal) => {
      try {
        const next = await getJob(jobId, logRef.current.length, signal);

        // The server echoes the cursor it actually honoured. If it resynced
        // us to 0 — a restarted job, a cursor past the end — the returned
        // slice is the whole log and replaces ours rather than appending to
        // it, which would otherwise duplicate every line.
        const since = next.log_since ?? 0;
        if (since === 0 && logRef.current.length > 0) {
          logRef.current = next.log;
        } else {
          logRef.current = logRef.current.concat(next.log);
        }
        setLog(logRef.current);

        // `log` on the returned object is only ever a delta; replacing it
        // with the accumulated log means consumers never see a partial one.
        setJob({ ...next, log: logRef.current });
        setNotFound(false);
        setError(null);

        if (isTerminal(next.status) && filesFetchedFor.current !== jobId) {
          filesFetchedFor.current = jobId;
          try {
            setFiles(await getJobFiles(jobId, signal));
          } catch {
            // Artifacts are shown from `job` as a fallback; a failed listing
            // is not worth blocking the whole screen over.
          }
        }
      } catch (e) {
        const status = (e as { status?: number }).status;
        if (status === 404) setNotFound(true);
        else if (status === 0) setError("cannot reach the dashboard server");
        else if (e instanceof Error) setError(e.message);
      } finally {
        setLoading(false);
      }
    },
    [jobId],
  );

  // Fast while the job is doing something, stopped once it cannot change
  // again. A finished job polled every two seconds forever is the exact
  // waste this rewrite exists to remove.
  const interval = job === null ? 1500 : isTerminal(job.status) ? null : 1500;

  usePoll(poll, { interval });

  return {
    job,
    log,
    files,
    loading,
    notFound,
    error,
    refresh: useCallback(() => setNonce((n) => n + 1), []),
  };
}
