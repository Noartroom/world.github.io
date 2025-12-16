// Service Worker for Immersive 3D Astro
// Cache Name: Versioned to force update on change
const CACHE_NAME = 'immersive-3d-v1';

// Assets to cache immediately (Core App Shell)
const STATIC_ASSETS = [
  '/',
  '/favicon.ico',
  '/favicon.svg',
  '/models/fallback-light.svg',
  '/models/fallback-dark.svg'
];

// Assets to cache lazily (Large 3D models, binaries)
// These are cached when requested, not on install
const DYNAMIC_ASSETS = [
    '/pkg/model_renderer_bg.wasm',
    '/pkg/model_renderer.js',
    '/models/newmesh.glb',
    '/models/dezimiertt-glb-03.glb'
];

self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting(); // Activate immediately
});

self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[Service Worker] Removing old cache', key);
            return caches.delete(key);
          }
        })
      );
    })
  );
  return self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // 1. Cache First for Assets (GLB, WASM, Images)
  if (event.request.url.includes('/models/') || 
      event.request.url.includes('/pkg/') || 
      event.request.url.endsWith('.svg') ||
      event.request.url.endsWith('.png')) {
      
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(event.request).then((response) => {
          // Check if valid response
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }
          // Clone and Cache
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
          return response;
        });
      })
    );
    return;
  }

  // 2. Network First for HTML (Navigations)
  // Ensures user always gets latest content, falls back to cache if offline
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          return caches.match(event.request);
        })
    );
    return;
  }

  // 3. Stale-While-Revalidate for CSS/JS
  // Serve cached version immediately, then update cache in background
  if (event.request.destination === 'style' || event.request.destination === 'script') {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        const fetchPromise = fetch(event.request).then((networkResponse) => {
           if (networkResponse.ok) {
               const clone = networkResponse.clone();
               caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
           }
           return networkResponse;
        });
        return cachedResponse || fetchPromise;
      })
    );
    return;
  }

  // Default: Network Only
  event.respondWith(fetch(event.request));
});
