import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        // In Docker Compose the API service is reachable via its service name.
        // Set PROXY_TARGET=http://api:9000 in the container environment to
        // override; defaults to localhost for plain local development.
        target: process.env.PROXY_TARGET || 'http://localhost:9000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
