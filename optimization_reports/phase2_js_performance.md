# Phase 2: JS/TS Performance Analysis

## 1. Render Loop Efficiency (`src/components/Renderer.astro`)

The Javascript render loop (`renderLoop`) drives the application. While it implements some "SOTA Guardrails" (FPS monitoring, Dynamic Resolution), it contains significant performance bottlenecks.

### Critical Issues
*   **Layout Thrashing (Forced Reflow):**
    *   **Location:** `applyUIRepulsion()` function called every frame.
    *   **Problem:** calls `document.querySelectorAll('.magnetic-ui')` AND `el.getBoundingClientRect()` inside the loop.
    *   **Impact:** Forces the browser to recalculate layout every single frame (16ms budget), drastically reducing time available for GPU submission and JS logic.
    *   **Recommendation:** Cache UI element references and their bounding boxes. Only update boxes on window resize.

*   **Garbage Collection (Memory Churn):**
    *   **Location:** `state.getBlobScreenPosition()`
    *   **Problem:** Returns a new JS Array `[x, y]` every frame from WASM.
    *   **Impact:** Creates 60-120 arrays per second, triggering frequent minor GC pauses.
    *   **Recommendation:** Use a shared `Float32Array` memory view (Wasm memory buffer) or pass a pre-allocated array to be filled by Rust.

## 2. Event Handler Analysis

*   **Pointer Events (`pointermove`):**
    *   **Current State:** Directly calls `state.update()` on every event. High-poll-rate mice (1000Hz) or smooth touchscreens can fire events faster than the frame rate.
    *   **Optimization:** Debounce/Throttle is not ideal for games/3D. **Recommendation:** Store input state in variables (`lastX`, `lastY`) and only call `state.update()` once per `renderLoop` tick.
    *   **Positive Note:** `cachedCanvasRect` optimization is already present, mitigating some reflows during input.

*   **Touch/Gestures:**
    *   Pinch-to-zoom logic is implemented in JS. This is fine, but ensuring `passive: false` is used correctly (it is) prevents scrolling interference.

## 3. Service Worker & Caching (`public/sw.js`)

*   **Strategy:** Cache-First for heavy assets (`.glb`, `.wasm`).
*   **Effectiveness:** Excellent for 3D web apps. Ensures subsequent loads are instant.
*   **Optimization Opportunity:** The cache name `immersive-3d-v4` is hardcoded. An automated versioning system during build would prevent stale caches during development iterations.

## 4. Nanostores Efficiency

*   **Subscriptions:** Subscriptions in `Renderer.astro` (`theme`, `activeModel`) are handled correctly with cleanup.
*   **Sync Issues:** The `onSet(theme, ...)` callback triggers a `requestAnimationFrame` to update lighting. This might desync by one frame relative to the CSS class update, but visually it should be negligible.

## 5. Prioritized Recommendations

1.  **Refactor `applyUIRepulsion`:**
    *   Cache `.magnetic-ui` elements on init.
    *   Cache their positions (`rect`) and only update on `resize` observer events.
    *   Use CSS Transforms (`translate3d`) for movement (already doing this, which is good).

2.  **Optimize WASM Interop:**
    *   Change `getBlobScreenPosition` to write to a shared Float32Array or return a packed f64 (if precision allows) to avoid object allocation.

3.  **Decouple Input from Logic:**
    *   Move `state.update()` calls from `pointermove` to `renderLoop`. Use event listeners only to update local coordinate state variables.

4.  **Static Object Caching:**
    *   Move `document.querySelectorAll` out of the loop!
