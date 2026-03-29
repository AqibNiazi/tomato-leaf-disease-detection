import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    // This proxy only applies to `npm run dev` — it is ignored in
    // the Vercel build. In production, VITE_API_BASE_URL on Vercel
    // points directly to the HF Space.
    proxy: {
      "/api": {
        target: "https://aqibniazi-tomato-disease-api.hf.space",
        changeOrigin: true,
        secure: true,
      },
    },
  },
});
