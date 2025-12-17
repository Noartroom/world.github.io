# Phase 4: Mobile-Specific Optimization

## 1. Mobile Rendering Strategy

*   **MSAA Config:** Currently, `start_renderer` reduces MSAA to 2x for mobile. This is a good baseline.
*   **Resolution Scaling:**
    *   **Issue:** High DPI mobile screens (e.g., iPhone Super Retina) have `devicePixelRatio` of 3.0. Rendering a 3D scene at native 3x resolution is extremely taxing and drains battery rapidly.
    *   **Recommendation:** Cap `maxDpr` to 2.0 (or even 1.5) specifically for mobile devices. The visual difference is negligible on small screens, but performance gains are massive.

## 2. Touch Input Performance

*   **Responsiveness:**
    *   Touch gestures (Pinch-to-zoom) are handled in `Renderer.astro` with `evCache`. This is standard and works well.
    *   **Issue:** `pointermove` fires very rapidly on modern touchscreens.
    *   **Recommendation:** Throttle input updates to the `requestAnimationFrame` loop (as mentioned in Phase 2) to prevent clogging the main thread.

## 3. Battery Consumption

*   **Idle Drain:**
    *   **Critical Issue:** The render loop runs at full 60 FPS even when the user is idle.
    *   **Impact:** Rapid battery drain even if the user is just looking at the screen without interacting.
    *   **Recommendation:** Connect the "User Idle" logic from `index.astro` (which adds `.user-idle` class) to the Renderer. When idle:
        1.  Reduce FPS to 30 or 15.
        2.  Or pause rendering completely if the scene is static (no auto-rotation).

## 4. Thermal Throttling

*   **Risk:** Continuous high-load rendering (Post-processing + high-poly model) will cause phones to heat up and throttle CPU/GPU speeds, leading to frame drops.
*   **Mitigation:** The "Dynamic Resolution Scaling" in `Renderer.astro` helps, but it reacts *after* frame drops occur. A proactive "Battery Saver" mode (toggleable or auto-enabled on low battery) that locks FPS to 30 and disables post-processing (Bloom/Tone Mapping) would be SOTA.

# Phase 5: Network & Asset Optimization

## 1. Asset Audit

*   **File Sizes:**
    *   `dezimiertt-glb-03.glb`: **30 MB** (Critical)
    *   `newmesh.glb`: **21 MB** (Critical)
    *   `modernart1-sculpt-1.glb`: **14 MB** (High)
*   **Impact:**
    *   Huge initial download.
    *   High memory usage (both main memory and GPU memory).
    *   Long parsing time on main thread (freezes UI).
*   **Recommendations:**
    1.  **Mesh Compression:** Use **Draco Compression**. It can reduce geometry size by 90%+. Requires client-side decoder (WASM), but `gltf` crate supports it via feature flags (or pre-decode in JS worker).
    2.  **Texture Optimization:** Ensure textures inside GLB are JPEG/WebP, not raw PNG/BMP. Resize textures to max 2048x2048 (or 1024x1024 for mobile).
    3.  **Optimization Tool:** Run assets through `gltf-transform` or `gltf-pipeline` to deduplicate accessors and compress.

## 2. Service Worker Strategy

*   **Caching:**
    *   The Cache-First strategy works, but caching 65MB+ of 3D models takes a significant chunk of browser storage quota.
    *   **Recommendation:** If assets are optimized (e.g., down to <5MB each), the current strategy is fine. If not, consider "Network First" for models to avoid filling cache with stale large files, or implement a cache expiration policy.

## 3. Delivery

*   **Compression:** Ensure the server serves `.glb` and `.wasm` files with `brotli` or `gzip` compression. This is often a server config, not code, but critical for transfer speed.
