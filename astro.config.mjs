// @ts-check
import { defineConfig } from 'astro/config';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// https://astro.build/config
export default defineConfig({
  vite: {
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
      },
    },
  },
});