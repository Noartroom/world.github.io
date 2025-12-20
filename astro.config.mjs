// @ts-check
import { defineConfig } from 'astro/config';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import AstroPWA from '@vite-pwa/astro';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/** @type {import('vite').Plugin} */
const wasmPlugin = {
  name: 'vite-plugin-wasm-env',
  enforce: 'pre',
  resolveId(id) {
    if (id === 'env') {
      return id;
    }
  },
  load(id) {
    if (id === 'env') {
      return `
        export function now() {
          return Date.now();
        }
      `;
    }
  },
};

// https://astro.build/config Remove? GH pages no longer in use
export default defineConfig({
  site: 'https://noartroom.github.io',
  vite: {
    plugins: [wasmPlugin],
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
      },
    },
    build: {
      rollupOptions: {
        external: ['/pkg/model_renderer_gl.js'],
      },
    },
  },
  integrations: [
    AstroPWA({
      strategies: 'injectManifest', // <--- KEEPS CUSTOM LOGIC
      srcDir: 'src',
      filename: 'sw.js',
      registerType: 'autoUpdate',
      injectManifest: {
        globPatterns: ['**/*.{js,css,html,svg,png,glb,wasm}'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
      }
    })
  ]
});
