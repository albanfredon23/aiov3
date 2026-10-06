import { resolve } from 'node:path';
import { defineConfig } from 'vite';

const page = (name) => resolve(import.meta.dirname, name);

export default defineConfig({
  build: {
    target: 'es2022',
    sourcemap: false,
    // Three.js (~145 ko gzip) est chargé à la demande, après le premier rendu.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      input: {
        index: page('index.html'),
        mentions: page('mentions-legales.html'),
        confidentialite: page('confidentialite.html'),
        cookies: page('cookies.html'),
      },
    },
  },
  server: {
    // En développement, l'API FastAPI tourne sur le port 8000.
    proxy: { '/api': 'http://localhost:8000' },
  },
  preview: {
    proxy: { '/api': 'http://localhost:8000' },
  },
});
