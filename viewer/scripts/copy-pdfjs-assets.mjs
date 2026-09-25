// Copy the pdf.js worker and character maps next to the built viewer. They must
// come from the pdfjs-dist version react-pdf resolved, or pdf.js refuses the
// worker.
import { cpSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const reactPdfDir = dirname(require.resolve("react-pdf/package.json"));
const pdfjsDir = dirname(
  createRequire(join(reactPdfDir, "package.json")).resolve("pdfjs-dist/package.json"),
);
const target = join("dist", "pdfjs");
mkdirSync(target, { recursive: true });
cpSync(join(pdfjsDir, "build", "pdf.worker.min.mjs"), join(target, "pdf.worker.min.mjs"));
cpSync(join(pdfjsDir, "cmaps"), join(target, "cmaps"), { recursive: true });
console.log(`Copied pdf.js worker and cmaps from ${pdfjsDir}`);
