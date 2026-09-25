import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Relative asset paths: each report serves its own copy of the viewer from
// report_X/viewer/, behind PanelApp's report proxy.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
});
