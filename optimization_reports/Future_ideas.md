# Design Doc: Procedural Generation & Crypto Visualization
**Engine Stack:** Rust, WGPU, WGSL, Glam
**Context:** AstroJS website with custom 3D renderer

---

## Part 1: Visualizing Cryptography (The "Logic" Layer)

### 1. The "Avalanche Effect" Soft Body
**Concept:** Visualize how a small change in input (1 bit) drastically changes the output (50% bits) using a hash function.
**Visual Style:** A floating, jelly-like sphere that ripples violently upon data input.

* **Implementation Strategy:**
    1.  **Geometry:** Generate a high-resolution Icosphere in Rust using `glam::Vec3`.
    2.  **Physics (Compute Shader):** Implement a Mass-Spring system. Each vertex is a "mass" connected to neighbors by "springs."
    3.  **The Trigger:**
        * *Suggestion:* When user input changes, calculate the SHA-256 hash in Rust.
        * *Suggestion:* Compare the new hash to the old hash. For every bit difference, apply an impulse force to a corresponding vertex in the `StorageBuffer`.

* **WGPU/WGSL Specifics:**
    * **Storage Buffers:** Use two buffers for vertex positions (`read_only` and `read_write`) to handle the compute step without race conditions (Ping-Pong buffers).
    * **Workgroups:** Map one thread per vertex.
    * **Performance Note:** Ensure your `struct Particle` in WGSL respects 16-byte alignment (std140/std430 layout) to avoid padding issues.

### 2. Elliptic Curve "Flow Fields"
**Concept:** Visualize the mathematical "flow" of Elliptic Curve Cryptography ($y^2 = x^3 + ax + b$).
**Visual Style:** A river of thousands of instanced particles following the invisible tangent lines of the curve.

* **Implementation Strategy:**
    1.  **Instancing:** Create a single mesh (e.g., a simple quad or low-poly arrow). Use WGPU Instancing to draw 10k+ copies.
    2.  **Flow Logic (Vertex/Compute Shader):**
        * Instead of calculating position on CPU, pass the curve parameters ($a, b$) as Uniforms.
        * *Suggestion:* In the shader, calculate the derivative of the curve at the particle's current position to determine velocity vectors.
    3.  **Interaction:** Slowly animate parameters $a$ and $b$ to show how the "security curve" shifts.

* **WGPU/WGSL Specifics:**
    * **Indirect Draw:** Consider `draw_indirect` if you want to dynamically change particle counts without rebuilding pipelines.
    * **Efficiency:** Use a "wrapping" logic in the shader. If a particle flows off-screen, reset its position to the opposite side to maintain density without spawning new entities.

### 3. Merkle Tree "Coral"
**Concept:** Visualize the verification path of a Merkle Tree.
**Visual Style:** Organic, branching coral structure where pulses travel from leaf (data) to root.

* **Implementation Strategy:**
    1.  **Rendering Technique:** Raymarching (SDFs) is superior here to standard geometry for smooth, organic branching.
    2.  **Data Structure:**
        * Flatten your Merkle Tree into a linear `StorageBuffer` (array representation of a binary tree).
    3.  **Shader Logic:**
        * *Suggestion:* Use smooth-min (`smin` in WGSL) to blend branches seamlessly.
        * *Suggestion:* Pass a "highlight path" index array. The shader increases the emission intensity for segments belonging to that path.

* **WGPU/WGSL Specifics:**
    * **Fragment Shader:** This is pixel-heavy. Render the coral to a smaller texture (off-screen render target) and upscale it if FPS drops.
    * **Bounding Box:** Raymarching the whole screen is expensive. Render a proxy cube geometry and only raymarch inside the fragment shader of that cube.

---

## Part 2: Procedural Art & Immersion (The "Vibe" Layer)

### 4. HTML-Aware Fluid Simulation
**Concept:** Background fluid that reacts to the HTML elements floating above it.
**Visual Style:** Ethereal smoke or liquid that flows *around* your website's UI.

* **Implementation Strategy:**
    1.  **The HTML Bridge:**
        * *Suggestion:* JavaScript (Astro) observes DOM element positions (`getBoundingClientRect`).
        * Send these coordinates to Rust -> WGPU Uniform Buffer (`array<vec4<f32>, N>` representing bounding boxes).
    2.  **Simulation:**
        * Use a Grid-based fluid solver (stable fluids) or SPH in a Compute Shader.
    3.  **Interaction:**
        * In the velocity update step of the fluid sim, add a repulsion vector away from the center of any active HTML bounding box.

* **WGPU/WGSL Specifics:**
    * **Texture Sharing:** Use a `StorageTexture` for the fluid density field. `textureStore` is very fast for this.
    * **Performance:** Run the physics update at a lower frequency (e.g., 30hz) than the render loop (60hz+), interpolating between states if necessary.

### 5. Raymarched "Living" Backgrounds
**Concept:** An abstract, breathing noise field that fills the void.
**Visual Style:** Giger-esque or cellular walls that undulate.

* **Implementation Strategy:**
    1.  **Noise:** Implement 3D Simplex or Perlin noise in WGSL.
    2.  **Time Domain:**
        * *Suggestion:* Feed `time` into the 4th dimension of the noise function. This creates non-repetitive, evolving movement without "sliding" textures.
    3.  **Depth Perception:**
        * Use User Mouse Position (passed via Uniforms) to slightly rotate the camera or skew the ray directions, creating a parallax effect.

* **WGPU/WGSL Specifics:**
    * **Full Screen Triangle:** Render a single triangle that covers the screen (vertices: `(-1, -1), (3, -1), (-1, 3)`) to trigger the fragment shader for the background without vertex overhead.

### 6. Crypto-Seeded Terrain
**Concept:** A unique world generated deterministically from a cryptographic seed (User ID / Session Key).
**Visual Style:** Low-poly landscapes or abstract topological maps.

* **Implementation Strategy:**
    1.  **Seeding (Rust Side):**
        * Use `rand_chacha` (ChaCha20 RNG) seeded with the user's unique crypto-key.
        * Generate the noise map or heightmap data on the CPU once at startup (or in chunks).
    2.  **Upload:**
        * Upload this height data to a WGPU texture or vertex buffer.
    3.  **Styling:**
        * *Suggestion:* Use a "wireframe" shader with barycentric coordinates to give it a technical, "blueprint" look that fits a crypto theme.

* **WGPU/WGSL Specifics:**
    * **LOD (Level of Detail):** If the terrain is large, implement a basic continuous LOD or quad-tree system to reduce vertex count for distant geometry.

---

## Technical Summary for Implementation

### Recommended Crate Additions
* `bytemuck`: Essential for casting Rust structs to `&[u8]` for WGPU buffers.
* `rand_chacha`: For cryptographically secure, reproducible procedural generation.
* `wgpu-profiler`: To measure the cost of your new Compute Passes.

### General Performance Guidelines
1.  **Minimize CPU-GPU Traffic:**
    * Calculate physics on GPU.
    * Only upload small data (mouse pos, HTML rects, user input strings) per frame.
    * Avoid `buffer.slice(..).map_async()` (reading back to CPU) during the render loop.
2.  **Pipeline Organization:**
    * **Init:** Create buffers/textures.
    * **Update:** Write uniforms (queue.write_buffer).
    * **Compute Pass:** Run simulation (Softbody / Fluid).
    * **Render Pass:** Draw geometry (Scene + Art).
3.  **WGSL Padding:**
    * Remember: `vec3<f32>` has the same alignment requirement as `vec4<f32>` (16 bytes) in Uniform/Storage buffers. Use `vec4` with a dummy value or explicit padding fields in Rust structs.

## Code-specific implementation recommendations

# Design Doc: Procedural Crypto-Visuals (v2)
**Context:** Rust/WGPU engine with existing PBR & Blob render pipelines.
**Current Stack:** `wgpu 0.20`, `glam`, `bytemuck`, `flume` (async assets).

---

## 🛠 Prerequisites: Engine Upgrades
Before implementing specific visuals, your `lib.rs` needs two specific upgrades to support animation and procedural data.

### 1. Add Global Time
Your current `SceneUniform` contains Camera, Light, and Blob data, but no `time`. Procedural animation requires a time delta.
* **Action:** Update `SceneUniform` in `lib.rs`.
    ```rust
    // lib.rs
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct SceneUniform {
        camera: CameraUniform,
        light: LightUniform,
        blob: BlobUniform,
        time: f32,       // <--- Add this
        _padding: [f32; 3], // Maintain 16-byte alignment
    }
    ```
* **Action:** Update `State::render` to pass `performance.now() / 1000.0` from JS to Rust, and write it to the buffer.

### 2. Enable Compute Shaders (Optional but Recommended)
Your current setup uses only `RenderPipeline`. For advanced particle effects (Idea #2), a `ComputePipeline` is efficient.
* **Action:** Add `compute_pipeline: wgpu::ComputePipeline` to your `State` struct if you choose the "Compute" path below.

---

## Part A: Visualizing Cryptography (Logic Layers)

### 1. The "Avalanche Effect" Soft Body (Refined)
**Concept:** Instead of a complex physics simulation, use **Vertex Displacement** in your existing `vs_blob` shader. This fits your current architecture perfectly without needing a new physics engine.
* **Visual:** The "Blob" (currently a sphere mesh) spikes and ripples when data changes.
* **Implementation (`lib.rs`):**
    1.  **Reuse `blob_pipeline`:** You already have `vs_blob`.
    2.  **Uniforms:** Add a `hash_entropy: f32` field to `BlobUniform`.
    3.  **Shader (`shader.wgsl`):** In `vs_blob`, use the `hash_entropy` to modulate the frequency and amplitude of a sine wave displacing the `model_position`.
    ```wgsl
    // Pseudo-code for vs_blob
    let noise = sin(vertex_pos * 10.0 + time + hash_entropy);
    let displacement = normal * noise * hash_entropy; // Only spikes when entropy > 0
    let final_pos = model_pos + displacement;
    ```
* **Trigger:** In `Renderer.astro`, when a user types, calculate a simple float from the input hash and pass it to `state.set_blob_entropy(val)`.

### 2. Elliptic Curve Flow Field (Instancing)
**Concept:** Visualize $y^2 = x^3 + ax + b$ as a flow field.
* **Implementation (`lib.rs`):**
    1.  **Instancing:** You currently draw meshes with `draw_indexed`. For this, create a new function `create_flow_particles` that generates a single "Arrow" or "Dot" mesh.
    2.  **Instance Buffer:** Create a `wgpu::Buffer` containing 10,000+ `InstanceRaw` structs (initial positions).
    3.  **Draw Call:** Use `render_pass.draw_indexed(..., 0..10000)` (instanced draw).
    4.  **Math:** Calculate the curve flow in the **Vertex Shader**.
        * Pass $a$ and $b$ curve parameters via `SceneUniform`.
        * Update particle positions based on the vector field defined by the curve derivative.
    * **Optimization:** Use the `time` uniform to offset positions so they loop, avoiding the need to update the buffer from the CPU every frame.

### 3. Merkle Tree "Raymarched Coral"
**Concept:** A fractal structure visualizing hash trees.
* **Implementation:**
    1.  **Technique:** Raymarching (SDF) inside a Fragment Shader.
    2.  **Proxy Geometry:** Draw a simple Cube (or your existing Sphere) that encompasses the area where the coral should be.
    3.  **Shader:** Inside `fs_model` (or a new `fs_coral`), instead of sampling a texture, run a raymarching loop.
    ```rust
    // lib.rs
    // Create a simple cube mesh to act as the "canvas" for the 3D fractal
    let proxy_mesh = create_cube_mesh(...);
    ```
    4.  **Data:** Pass the "Merkle Root" hash as a uniform color or seed value to change the fractal's branching parameters.

---

## Part B: Procedural Art & Immersion (Vibe Layers)

### 4. HTML-Repulsive Background (Inverse of Current)
**Concept:** Currently, your `Renderer.astro` pushes *HTML* away from the *Blob*. Invert this: make the *background fog/particles* pushed away by *HTML*.
* **Implementation (`lib.rs`):**
    1.  **New Uniform:** `HtmlRectsUniform`.
        ```rust
        struct HtmlRectsUniform {
            rects: [[f32; 4]; 16], // Support up to 16 UI elements (x, y, w, h)
            count: u32,
        }
        ```
    2.  **Update Loop:** In `Renderer.astro`, query `.magnetic-ui` elements and send their normalized screen coordinates to Rust.
    3.  **Shader:** In your background shader (or a fog post-process pass), check the distance of the pixel to these rects.
        * If `distance < threshold`, reduce fog density or change color.

### 5. "Living" Noise Terrain (Texture Generation)
**Concept:** Generate a unique terrain texture on the CPU using `rand_chacha` and your async texture loader.
* **Implementation (`lib.rs`):**
    1.  **Texture Gen:** Inside `start_renderer` (or a new async task), generate a `Vec<u8>` representing Perlin noise. Seed the RNG with the user's session ID.
    2.  **Reuse `Texture::from_bitmap`:** You already have this! Wrap your generated bytes in an `ImageData` or `ImageBitmap` (via `web-sys`) or upload raw bytes directly using `queue.write_texture`.
    3.  **Mesh:** Create a high-vertex-count Plane mesh.
    4.  **Displacement:** In `vs_model`, sample this generated texture to displace vertices up/down (Heightmap).

---

## 🚀 Recommended Implementation Plan (Priority Order)

1.  **The "Low Hanging Fruit" - Animated Blob (Idea #1)**
    * **Why:** You already have the `blob_pipeline`, `vs_blob`, and `SceneUniform`.
    * **Task:** Add `time` to `SceneUniform` and modify `vs_blob` in `shader.wgsl` to add sine-wave vertex displacement.
    * **Cost:** Low. No new Rust structs needed, just shader code and 1 uniform update.

2.  **The "Background Vibe" - Raymarched Proxy (Idea #3)**
    * **Why:** Adds high visual impact without complex geometry management.
    * **Task:** Create a `create_cube_mesh` helper. Create a new pipeline `coral_pipeline` that uses a raymarching fragment shader.
    * **Cost:** Medium. Needs a new pipeline state in `lib.rs`.

3.  **The "Advanced Tech" - Instanced Flow Field (Idea #2)**
    * **Why:** Best visualization of actual crypto math (ECC).
    * **Task:** Requires setting up Instancing buffers (`wgpu::BufferUsages::VERTEX`) and a new pipeline.
    * **Cost:** High. Significant boilerplate in `lib.rs`.

### Performance Note for `lib.rs`
* **Buffer Writes:** You are using `queue.write_buffer` for uniforms. This is good. For particle systems (Idea #2), avoid writing to buffers every frame from the CPU. Use `time` in the shader to animate static buffers, OR use a Compute Shader to update positions entirely on the GPU.
* **WASM Boundary:** Minimize calls across JS/WASM. Your current `update()` method is good, but try to batch the HTML Rect updates (Idea #4) into a single array view rather than individual calls.