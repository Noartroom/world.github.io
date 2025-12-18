// Service Worker for Immersive 3D Astro
// Cache Name: Versioned to force update on change
const CACHE_NAME = 'immersive-3d-v9'; // TODO: THis should be updated dynamically e.g. on each file change. currently i have to increase this manually.

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

  // --- STRATEGY 2: Network First (HTML / Navigation) ---
  // Ensure we always get the latest index.html so we don't request 404 assets
  if (e.request.mode === 'navigate' || url.pathname === '/') {
    e.respondWith(
      fetch(e.request)
        .then((networkResp) => {
          return caches.open(CACHE_NAME).then((cache) => {
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
      caches.open(CACHE_NAME).then((cache) => {
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
