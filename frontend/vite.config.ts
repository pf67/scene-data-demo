import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [vue()],
  base: "/",
  optimizeDeps: { exclude: ["@sparkjsdev/spark"] },
  server: { proxy: { "/api": "http://127.0.0.1:8090" } },
  build: {
    outDir: "../web",
    emptyOutDir: true,
    rollupOptions: { input: resolve(import.meta.dirname, "scene-standalone.html") },
  },
});
