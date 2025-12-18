// Service Worker for Immersive 3D Astro

import { precacheAndRoute, cleanupOutdatedCaches } from 'workbox-precaching';
import { clientsClaim, skipWaiting } from 'workbox-core';
import { registerRoute } from 'workbox-routing';
import { NetworkFirst, CacheFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';

// --- 1. WORKBOX LOGIC (Handles Pre-cached Assets) ---
// // Takes over the client immediately (like self.skipWaiting)
skipWaiting();
// Controls clients immediately after activation
clientsClaim();

// 2. Cache Cleanup (Replaces your manual cache deletion loop)
// Automatically deletes old precaches when a new version installs
cleanupOutdatedCaches();

// 3. Pre-Caching (Replaces STATIC_ASSETS array)
// The plugin injects the file list here automatically.
// This handles caching '/', '/manifest.json', and your fallbacks.
precacheAndRoute(self.__WB_MANIFEST);

// 1. 3D Assets: Cache First (Replaces Strategy 1)
registerRoute(
  ({ url }) => url.pathname.match(/\.(glb|gltf|bin|wasm|png|jpg|jpeg|webp|svg)$/),
  new CacheFirst({
    cacheName: 'immersive-3d-assets', // This replaces RUNTIME_CACHE_NAME
    plugins: [
      new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 30 * 24 * 60 * 60 }),
    ],
  })
);

// 2. HTML/Navigation: Network First (Replaces Strategy 2)
// This overrides the default CacheFirst precache behavior for navigation
registerRoute(
  ({ request }) => request.mode === 'navigate',
  new NetworkFirst({
    cacheName: 'immersive-pages',
    networkTimeoutSeconds: 3, // Fallback to cache if network takes > 3s
  })
);

// 3. CSS/JS: Stale While Revalidate (Replaces Strategy 3)
registerRoute(
  ({ request }) => request.destination === 'script' || request.destination === 'style',
  new StaleWhileRevalidate({
    cacheName: 'immersive-static-resources',
  })
);
