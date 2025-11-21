// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  integrations: [
    react({
      // This is the critical line from the documentation you sent.
      // It tells Astro to treat all .tsx files as React components.
      include: ['**/*.tsx'],
    }),
  ]
});