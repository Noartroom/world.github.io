# Phase 6: Accessibility Analysis

## 1. Compliance Audit

*   **Layer Management:** The app correctly uses `inert` on the inactive `layer-back` and `layer-front` to remove them from the accessibility tree. This is SOTA practice.
*   **Controls:** Buttons have `aria-label` and `aria-pressed`. `role="slider"` is used for the time control.
*   **Canvas:** The `<canvas>` element has `tabindex="0"` and `aria-label="3D Canvas"`, making it focusable and identifiable.

## 2. Motion Sensitivity (`prefers-reduced-motion`)

*   **Detection:** The app detects this preference in `deviceStore.ts`.
*   **Implementation Gap:** While detected, this flag is **ignored** by the 3D renderer. The camera continues to move/drift, and smooth transitions (lerping) are not disabled.
*   **Recommendation:**
    *   Expose a `setReducedMotion(bool)` method on the WASM `State`.
    *   If true, disable camera inertia, auto-rotation, and use instant transitions instead of lerping.

## 3. Keyboard Navigation

*   **Status:** The Time Control supports Arrow Keys (`ArrowUp`/`ArrowDown`).
*   **Missing:** The 3D Camera itself does not seem to support keyboard navigation (VASD/Arrows), only mouse/touch drag.
*   **Recommendation:** Map Arrow Keys to camera rotation in `Renderer.astro` when the canvas has focus.

# Phase 7: Code Quality & Security

## 1. Code Organization ("God File" Issue)

*   **Issue:** `src/lib.rs` contains **1959 lines** of code. It mixes:
    *   State Management
    *   Rendering Logic (Pipelines, Passes)
    *   Model Loading (GLTF parsing)
    *   Mesh Generation
    *   Uniform Definitions
*   **Impact:** Poor maintainability, difficult to test, hard to onboard new developers.
*   **Recommendation:** Refactor into modules:
    *   `renderer.rs`: WGPU setup, pipelines, render pass.
    *   `model.rs`: Mesh, Texture, Model structs and loading logic.
    *   `state.rs`: The main `State` struct and public API.
    *   `uniforms.rs`: Uniform structs (`SceneUniform`, etc.).

## 2. Type Safety

*   **Issue:** `Renderer.astro` uses `// @ts-ignore` for the WASM import: `import init, { startRenderer } from '/pkg/model_renderer.js';`.
*   **Risk:** Runtime errors if the WASM signature changes.
*   **Recommendation:** Generate TypeScript definitions (`.d.ts`) using `wasm-pack` with the `--typescript` flag (it's currently disabled in `package.json` script: `--no-typescript`).

## 3. Security

*   **GLB Parsing:** The manual byte-patching in `load_model_from_bytes` (`copy_from_slice`) is risky if not strictly bounds-checked. While unlikely to be exploited if assets are self-hosted, it's a potential panic vector.
*   **Dependencies:** `wgpu` 0.20 is up-to-date and secure.
