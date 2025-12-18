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

const RUNTIME_CACHE_NAME = 'immersive-3d-runtime-v1';

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // CRITICAL WARNING:
  // If Workbox has already precached a file (like index.html or style.css),
  // it usually serves it "Cache First" immediately. Your listeners below 
  // might NOT run for those files. See the "Order of Operations" note below.
  // --- STRATEGY 1: Cache First (Heavy 3D Assets & Images) ---
  // Matches: .glb, .wasm (including your ?v=gpu_fix_6), images
  if (url.pathname.match(/\.(glb|gltf|bin|wasm|png|jpg|jpeg|webp|svg)$/)) {
    e.respondWith(
      caches.open(RUNTIME_CACHE_NAME).then((cache) => {
        return cache.match(e.request).then((cachedResp) => {
          // Return cache if found
          if (cachedResp) return cachedResp;

          // Else fetch from network
          return fetch(e.request).then((networkResp) => {
            // Cache valid responses for next time
            if (networkResp.ok) {
              cache.put(e.request, networkResp.clone());
            }
            return networkResp;
          });
        });
      })
    );
    return;
  }

  // --- STRATEGY 2: Network First (HTML / Navigation) ---
  // NOTE: If index.html is in __WB_MANIFEST, Workbox might handle this before we get here!
  // Ensure we always get the latest index.html so we don't request 404 assets
  if (e.request.mode === 'navigate' || url.pathname === '/') {
    e.respondWith(
      fetch(e.request)
        .then((networkResp) => {
          return caches.open(RUNTIME_CACHE_NAME).then((cache) => {
            cache.put(e.request, networkResp.clone());
            return networkResp;
          });
        })
        .catch(() => {
          return caches.match(e.request); // Offline fallback
        })
    );
    return;
  }

  // --- STRATEGY 3: Stale-While-Revalidate (CSS / JS) ---
  // For styles and scripts, try cache first, but update in background
  if (e.request.destination === 'script' || 
      e.request.destination === 'style') {
    
    e.respondWith(
      caches.open(RUNTIME_CACHE_NAME).then((cache) => {
        return cache.match(e.request).then((cachedResp) => {
          const fetchPromise = fetch(e.request).then((networkResp) => {
             if (networkResp.ok) {
               cache.put(e.request, networkResp.clone());
             }
             return networkResp;
          }).catch(() => {
             // Swallow offline errors
          });
          return cachedResp || fetchPromise;
        });
      })
    );
    return;
  }
});
