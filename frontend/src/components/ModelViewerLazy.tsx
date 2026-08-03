import React, { Suspense } from "react";

const ModelViewer = React.lazy(() => import("./ModelViewer"));

export function ModelViewerLazy(props: {
  url: string;
  className?: string;
}): React.JSX.Element {
  return (
    <Suspense
      fallback={
        <div
          className={`flex items-center justify-center bg-void font-mono text-xs text-ink-faint ${
            props.className ?? ""
          }`}
          aria-live="polite"
        >
          loading viewer…
        </div>
      }
    >
      <ModelViewer {...props} />
    </Suspense>
  );
}
