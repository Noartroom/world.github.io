// Service Worker for Immersive 3D Astro
// Cache Name: Versioned to force update on change
const CACHE_NAME = 'immersive-3d-v2';

// Files to pre-cache immediately so the "Skeleton" works offline
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/models/fallback-light.svg', // Vital for offline fallback
  '/models/fallback-dark.svg'
];

self.addEventListener('install', (e) => {
  self.skipWaiting(); // Take over immediately
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Pre-caching Static Shell');
      return cache.addAll(STATIC_ASSETS);
    })
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(keys.map((key) => {
        if (key !== CACHE_NAME) return caches.delete(key);
      }));
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // --- STRATEGY 1: Cache First (Heavy 3D Assets & Images) ---
  // Matches: .glb, .wasm (including your ?v=gpu_fix_6), images
  if (url.pathname.match(/\.(glb|gltf|bin|wasm|png|jpg|jpeg|webp|svg)$/)) {
    e.respondWith(
      caches.open(CACHE_NAME).then((cache) => {
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

  // --- STRATEGY 2: Stale-While-Revalidate (App Shell / Logic) ---
  // For HTML, JS, CSS. Loads instantly from cache, updates in background.
  if (e.request.destination === 'document' || 
      e.request.destination === 'script' || 
      e.request.destination === 'style' ||
      url.pathname === '/') {
    
    e.respondWith(
      caches.open(CACHE_NAME).then((cache) => {
        return cache.match(e.request).then((cachedResp) => {
          
          const fetchPromise = fetch(e.request).then((networkResp) => {
             if (networkResp.ok) {
               cache.put(e.request, networkResp.clone());
             }
             return networkResp;
          }).catch(() => {
             // Swallow offline errors if we have a cache
          });

          return cachedResp || fetchPromise;
        });
      })
    );
    return;
  }
});

