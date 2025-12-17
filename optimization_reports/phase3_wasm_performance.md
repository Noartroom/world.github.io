# Phase 3: Rust/WASM Performance Analysis

## 1. WASM Binary Size Optimization

*   **Current State:** The `Cargo.toml` lacks a `[profile.release]` section, meaning the build uses default release settings. `console_error_panic_hook` is enabled by default in the `features`.
*   **Impact:** Larger than necessary WASM file, leading to slower download and hydration.
*   **Recommendations:**
    *   Enable **Link Time Optimization (LTO)**: `lto = true`
    *   Optimize for Size: `opt-level = 'z'` (or 's')
    *   Strip Symbols: `strip = true`
    *   Panic Handler: Use `panic = "abort"` to remove stack unwinding code.
    *   Feature Flag: Make `console_error_panic_hook` optional and disable it for production builds.

## 2. Render Pipeline & Resource Management

*   **Mesh Batching (Critical):**
    *   **Issue:** `State::load_model_from_bytes` creates a separate `VertexBuffer`, `IndexBuffer`, and `BindGroup` for *every mesh primitive* in the GLB.
    *   **Impact:** High number of draw calls and pipeline state changes (`set_vertex_buffer`, `set_bind_group`). This is a bottleneck for complex models.
    *   **Recommendation:** Implement **Geometry Batching**. Merge all static geometry into a single large Vertex/Index buffer. Use `draw_indexed` with `base_vertex` and `first_index` offsets to draw specific parts. This reduces buffer switching to near zero.

*   **Uniform Updates:**
    *   **Issue:** `update_scene_uniforms` writes to the GPU buffer every frame via `queue.write_buffer`.
    *   **Impact:** Unnecessary bus traffic if the camera/light hasn't changed.
    *   **Recommendation:** Add a `dirty` flag in the `State` struct. Only write to the buffer if `camera` moved, `light` changed, or `blob` moved.

*   **Transparency Sorting:**
    *   **Issue:** `transparent_to_draw.sort_by` runs on the CPU every frame.
    *   **Impact:** CPU overhead on the main thread.
    *   **Recommendation:** Throttle sorting. Only re-sort if the camera angle changes by a certain threshold (e.g., > 1 degree).

## 3. Shader Complexity (`src/shader.wgsl`)

*   **PBR Implementation:**
    *   The shader implements full PBR (GGX/Smith/Fresnel) and ACES Tone Mapping.
    *   **Optimization:** The ACES approximation involves multiple divisions and matrix-like ops per pixel.
    *   **Mobile Recommendation:** Use a simpler Tone Mapping (e.g., Reinhard or optimized ACES) and simpler PBR (Blinn-Phong) for "Low Power" mode.

*   **Lighting:**
    *   Point light attenuation uses `1.0 / (1.0 + ... dist^2)`.
    *   **Optimization:** This is generally fine, but could be pre-calculated or simplified for distant lights.

## 4. MSAA Strategy

*   **Current:** `const SAMPLE_COUNT: u32 = 4;` hardcoded.
*   **Mobile Issue:** 4x MSAA is very heavy for mobile GPUs (bandwidth intensive).
*   **Recommendation:**
    *   Expose `SAMPLE_COUNT` as a config parameter to `start_renderer`.
    *   Pass `2` or `1` (off) for mobile devices or based on `deviceStore` detection.

## 5. Summary of High Impact Changes

1.  **Add `[profile.release]` optimizations to `Cargo.toml`.** (Quick Win)
2.  **Implement `dirty` checking for Uniform updates.** (Easy)
3.  **Make MSAA configurable from JS.** (Medium)
4.  **Implement Geometry Batching.** (Hard, but highest performance gain for complex models)
