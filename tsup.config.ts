import { defineConfig } from "tsup";
import { cp } from "fs/promises";
import { join } from "path";

export default defineConfig([
  // Main library bundle
  {
    format: ["esm", "cjs"],
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
    format: ["esm", "cjs"],
    entry: ["src/models/*"],
    outDir: "dist/models/",
    dts: true,
    sourcemap: true,
    clean: true,
    target: "es2020",
    platform: "browser",
    skipNodeModulesBundle: false, // Bundle dependencies into worker
    noExternal: [
      "@lenml/tokenizer-gpt2",
      "onnxruntime-web",
      "idb-keyval",
      "@xenova/transformers",
    ], // Force bundle these packages
    outExtension: ({ format }) => ({
      js: format === "esm" ? ".mjs" : ".js",
    }),
    async onSuccess() {
      // Copy ONNX runtime files to dist/ort
      try {
        const sourcePath = join(
          process.cwd(),
          "node_modules",
          "onnxruntime-web",
          "dist"
        );
        const targetPath = join(process.cwd(), "dist/models");

        await cp(sourcePath, targetPath, {
          recursive: true,
        });
      } catch (error) {
        console.error("Error copying WASM files:", error);
      }
    },
  },
]);
