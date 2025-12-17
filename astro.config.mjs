// @ts-check
import { defineConfig } from 'astro/config';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

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

// https://astro.build/config
export default defineConfig({
  site: 'https://world.github.io',
  vite: {
    plugins: [wasmPlugin],
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
      },
    },
  },
});
