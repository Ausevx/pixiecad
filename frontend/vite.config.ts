import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The bundle is built straight into the Python package so the wheel ships it
// and FastAPI can mount it without a separate artefact step.
const OUT = "../src/pixiecad/web/static/dist";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/ui/",
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    outDir: OUT,
    emptyOutDir: true,
    // The dashboard is served from localhost, so the only thing a source map
    // costs is disk. Being able to read a stack trace off a real failure is
    // worth more than that.
    sourcemap: true,
    rollupOptions: {
      output: {
        // three.js is the single largest dependency and only the job detail
        // route needs it. Isolating it keeps the jobs list and new-job screens
        // from paying ~170 KB gzip they never use.
        manualChunks(id) {
          if (id.includes("node_modules/three")) return "three";
        },
      },
    },
  },
  server: {
    port: 5173,
    // `npm run dev` talks to a real dashboard on :8000 rather than a mock, so
    // the UI is always developed against the actual endpoint contract.
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
