import { execSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vitest/config";

// Stale-dist guard: every build writes dist/build-stamp.json (git hash +
// build time). The server logs it at startup and reports it in /diagnose,
// so a bundle that predates the source is visible instead of silently
// serving old recipe logic.
function buildStamp(): Plugin {
  return {
    name: "build-stamp",
    apply: "build",
    closeBundle() {
      let git = "unknown";
      try {
        git = execSync("git rev-parse --short HEAD").toString().trim();
      } catch {
        // shallow export or no git — the stamp still records the build time
      }
      const stamp = { git, built_at: new Date().toISOString() };
      writeFileSync(resolve(__dirname, "dist/build-stamp.json"), JSON.stringify(stamp));
    },
  };
}

// Dev server runs on 5173 (allowlisted by the API's origin policy) and talks
// to the API cross-origin at 127.0.0.1:8471. The production build is served
// by FastAPI itself, same-origin, from ui/dist.
export default defineConfig({
  plugins: [react(), buildStamp()],
  server: { port: 5173, strictPort: true },
  test: { environment: "node" },
});
