import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      "/jobs": `http://localhost:${process.env.API_PORT ?? "8000"}`,
      "/meshes": `http://localhost:${process.env.API_PORT ?? "8000"}`,
      "/health": `http://localhost:${process.env.API_PORT ?? "8000"}`,
    },
  },
})
