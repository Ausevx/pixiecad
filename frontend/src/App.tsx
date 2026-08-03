import { Shell } from "@/shell/Shell";
import { JobsRoute } from "@/routes/JobsRoute";
import { NewJobRoute } from "@/routes/NewJobRoute";
import { JobDetailRoute } from "@/routes/JobDetailRoute";
import { activeCount, useJobs } from "@/lib/jobs";
import { useRoute } from "@/lib/router";

export function App() {
  const route = useRoute();
  const { jobs } = useJobs();

  // The background field intensifies while work is in flight. One running job
  // is most of the way there; the difference between three and four is not
  // information anyone needs, so it saturates quickly.
  const active = activeCount(jobs);
  const activity = active === 0 ? 0 : Math.min(1, 0.6 + active * 0.2);

  return (
    <Shell activity={activity}>
      {route.kind === "jobs" && <JobsRoute />}
      {route.kind === "new" && <NewJobRoute />}
      {route.kind === "job" && <JobDetailRoute jobId={route.id} />}
    </Shell>
  );
}
