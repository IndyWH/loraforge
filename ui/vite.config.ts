import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev server runs on 5173 (allowlisted by the API's origin policy) and talks
// to the API cross-origin at 127.0.0.1:8471. The production build is served
// by FastAPI itself, same-origin, from ui/dist.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  test: { environment: "node" },
});
