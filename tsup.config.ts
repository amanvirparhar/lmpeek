import { defineConfig } from "tsup";

export default defineConfig([
  // Main library bundle
  {
    format: ["cjs", "esm"],
    entry: ["src/index.ts"],
    dts: true,
    shims: true,
    skipNodeModulesBundle: true,
    clean: true,
    target: "es2020",
    platform: "browser",
  },
  // Worker bundle
  {
    format: ["cjs", "esm"],
    entry: ["src/workers/gpt-2.ts"],
    dts: true,
    shims: true,
    clean: true,
    target: "es2020",
    platform: "browser",
    outExtension: () => ({ js: ".js" }),
  },
]);
