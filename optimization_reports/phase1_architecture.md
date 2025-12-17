# Phase 1: Architecture & Codebase Analysis

## 1. Architecture Overview

The application utilizes a **Hybrid Architecture** combining a static/SSR shell (Astro) with a high-performance native rendering core (Rust/WASM/WebGPU).

### Core Components
*   **Astro Shell (`src/pages/index.astro`):** Orchestrates the application lifecycle, manages the global UI overlay (Layers System), and handles initial routing.
*   **UI Layer (React/HTML):** Standard DOM elements overlaid on the canvas. Uses `inert` attribute for focus management between layers (Front/Back/Persistent).
*   **Rendering Core (`src/lib.rs`):** A custom WebGPU engine written in Rust.
    *   **WASM Bridge:** `wasm-bindgen` exposes the `State` struct and methods to JavaScript.
    *   **Interop:** Single-way control flow for the most part (JS -> Rust). Rust calls back to JS mainly for logging and async resource loading (Textures).

### Data Flow
1.  **User Input:** Captured in JS (`Renderer.astro`) via Pointer Events.
2.  **State Update (JS):** `sceneStore`, `deviceStore` update.
3.  **Bridge Call:** `Renderer.astro` calls `state.update()`, `state.setTheme()`, etc.
4.  **Rust State:** Updates internal structs (`CameraUniform`, `LightUniform`).
5.  **Render Loop:** `state.render()` called every animation frame (driven by JS `requestAnimationFrame`).

## 2. State Management Review

*   **Library:** `nanostores` (Standard/Persistent).
*   **Pattern:** Atomic state stores (`atom`, `map`).
*   **Efficiency:** High. Direct subscription in components ensures only necessary updates trigger side effects.
*   **Synchronization:**
    *   `theme` and `activeModel` are persistent, restoring user preference.
    *   Synchronization with WASM is manual inside `Renderer.astro` via `onSet` listeners.
    *   **Potential Issue:** Use of `requestAnimationFrame` inside a subscription callback (`themeUnsub`) in `Renderer.astro` to update Time-of-Day lighting. This creates a potential race condition or frame delay.

## 3. Rust/WASM Architecture

*   **Graphics Backend:** `wgpu` (WebGPU) with fallback logic in JS (SVG).
*   **Memory Strategy:**
    *   **Meshes:** Loaded via `gltf` crate. Vertex/Index buffers created on GPU.
    *   **Textures:** Asynchronous loading via `ImageBitmap` (Browser API) -> `queue.copy_external_image_to_texture`. This is the SOTA way to load textures in WebGPU.
    *   **Uniforms:** `SceneUniform` (Camera, Light, Blob) updated every frame via `queue.write_buffer`.
    *   **Transparency:** Manual sorting of transparent meshes every frame on CPU.

*   **Threading Model:**
    *   Single-threaded WASM main loop.
    *   Async tasks (image decoding) offloaded to browser thread pool via `wasm_bindgen_futures`.

## 4. Memory & Resource Management

*   **Lifecycle:** Explicit `state.free()` called on component unmount (`astro:before-swap`). This is critical for preventing GPU context leaks in SPA navigation.
*   **Assets:**
    *   GLB models are loaded fully into memory (`Vec<u8>`) before parsing.
    *   **Risk:** Large models could spike WASM memory usage. `modified_bytes = std::borrow::Cow::Owned(owned)` in `load_model_from_bytes` forces a copy if headers need patching.

## 5. Dependency Audit

*   **Rust:**
    *   `wgpu` (0.20): Modern, stable.
    *   `glam`: Standard math library.
    *   `flume`: High-performance channel. Good choice.
    *   `gltf`: Standard loader.
*   **NPM:**
    *   `@playcanvas/react`: Listed in `package.json` but **UNUSED** in the analyzed files. The renderer is custom WGPU. **Recommendation: Remove.**

## 6. Recommendations (No Code Changes Yet)

1.  **Optimize Interop:** Reduce frequency of JS<->Rust calls. The UI Repulsion logic (`getBlobScreenPosition`) generates garbage (arrays) every frame.
2.  **Remove Unused Deps:** Remove `@playcanvas/react`.
3.  **Memory:** Investigate streaming or zero-copy parsing for GLB if models grow larger.
4.  **Uniform Updates:** Check `update_scene_uniforms` in Rust. It runs every frame. If the camera/light hasn't changed, this is wasted bandwidth.
5.  **Transparency Sorting:** Optimization candidate. Only sort if camera angle changed significantly.
