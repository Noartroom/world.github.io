1. state.rs: Physics Hoisting & Zero-Copy Cache

This change prevents the CPU from recalculating the mouse ray multiple times per frame and prepares the data for JavaScript to read without allocation.

Rust
// state.rs
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
// ... other imports

// 1. New Helper Struct (Hoisted Ray)
pub struct WorldRay {
    pub origin: Vec3,
    pub direction: Vec3,
}

pub struct GameState {
    // ... existing fields ...
    
    // 2. Zero-Copy Cache Field
    // We store the result here so JS can just "peek" at the memory address
    pub blob_screen_pos_cached: [f32; 2], 
}

impl GameState {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, material_layout: &wgpu::BindGroupLayout) -> Self {
        // ... existing init ...
        Self {
            // ... existing fields ...
            blob_screen_pos_cached: [-1.0; 2], // Init off-screen
            // ...
        }
    }

    // 3. Optimized Update Loop
    pub fn update(&mut self, time: f64, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32, device: &wgpu::Device, material_layout: &wgpu::BindGroupLayout) {
        // ... [Existing Asset Loading Code] ...

        // 4. HOISTING: Calculate Ray ONCE per frame
        [cite_start]// Previously, this heavy matrix math ran inside the physics sub-steps [cite: 2]
        let mouse_ray = self.calculate_mouse_ray(mouse_x, mouse_y, screen_width, screen_height);

        // 5. Physics Loop (Cheaper now)
        let frame_time = time - self.last_time;
        self.last_time = time;
        self.accumulator += frame_time;
        
        // Safety cap for death spiral
        if self.accumulator > 0.25 { self.accumulator = 0.25; }

        const DT: f64 = 1.0 / 60.0;
        while self.accumulator >= DT {
            // Pass the pre-calculated ray reference
            self.step_physics(&mouse_ray, screen_width, screen_height, mouse_x, mouse_y); 
            self.accumulator -= DT;
        }

        // 6. Zero-Allocation Screen Position Calculation
        // Update the cached field that JS will read
        if self.blob_exists {
            let screen_pos = self.project_3d_to_screen(self.blob_position, screen_width, screen_height);
            self.blob_screen_pos_cached = screen_pos; 
            
            // Interpolation Logic
             let alpha = (self.accumulator / DT) as f32;
             let interpolated_pos = self.blob_prev_position.lerp(self.blob_position, alpha);
             
             self.scene_uniform.blob.position = [interpolated_pos.x, interpolated_pos.y, interpolated_pos.z, 0.0];
             self.blob_light_pos_3d = interpolated_pos;
        }
        
        // ... [Rest of update: lighting, camera uniforms] ...
    }

    // New Helper to calculate ray (Moved out of step_physics)
    fn calculate_mouse_ray(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> WorldRay {
        let mouse_norm_x = (mouse_x / screen_width) * 2.0 - 1.0;
        let mouse_norm_y = 1.0 - (mouse_y / screen_height) * 2.0;
        
        let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
        let y = self.camera_radius * self.camera_polar.cos();
        let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
        let cam_pos = vec3(x, y, z) + self.camera_target;

        let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), self.aspect_ratio, 0.1, 100.0);
        let inv_view_proj = (proj * view).inverse();

        let near_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 0.0, 1.0);
        let far_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 1.0, 1.0);
        let near_world = vec3(near_point.x, near_point.y, near_point.z) / near_point.w;
        let far_world = vec3(far_point.x, far_point.y, far_point.z) / far_point.w;

        WorldRay {
            origin: near_world,
            direction: (far_world - near_world).normalize(),
        }
    }

    // Updated step_physics signature
    fn step_physics(&mut self, ray: &WorldRay, screen_width: f32, screen_height: f32, mouse_x: f32, mouse_y: f32) {
        self.blob_prev_position = self.blob_position;

        // Optimized Cursor Light
        if self.cursor_light_active {
            let light_distance = 3.0;
            self.cursor_light_pos_3d = ray.origin + ray.direction * light_distance;
        }

        // Optimized Drag Logic (reusing ray)
        if self.blob_dragging && self.blob_exists {
            // Re-calculate drag depth specific logic here if needed, or use ray intersection
            // For now, we can adapt the existing drag logic to use the ray components:
            
            // ... [Keep existing drag depth logic or adapt to use ray.origin/direction] ...
            // Example adaptation:
            let depth = if self.blob_drag_depth > 0.0 { self.blob_drag_depth } else { 0.5 }; // Simplified
            // Note: You can keep the original full drag logic here if you prefer, 
            // but using the pre-calculated 'ray' struct avoids the 'inv_view_proj' recalc.
            
            let mouse_world = ray.origin + ray.direction * (depth * 10.0); // Simplified projection for example
            // ... [Rest of blob constraint logic] ...
             
             // To strictly preserve your exact drag behavior, you might need to pass the 'inv_view_proj' 
             // in the Ray struct or recalculate just the depth projection. 
             // Given the performance gain, it's worth simply recalculating the drag point 
             // using the ray origin + direction * distance_to_drag_plane.
        }
        
        // ... [Rest of smoothing logic] ...
        if self.blob_exists {
            let smoothing_factor = if self.blob_dragging { 0.8 } else { 0.15 };
            let diff = self.blob_target_position - self.blob_position;
            self.blob_position = self.blob_position + diff * smoothing_factor;
        }
    }
}
2. lib.rs: The Clean Interface

We expose raw numbers (f32) instead of objects.

Rust
// lib.rs
// ... imports

#[wasm_bindgen]
impl State {
    // ... existing methods ...

    // REPLACED: get_blob_screen_position()
    // New Zero-Copy Getters:
    #[wasm_bindgen(js_name = "getBlobScreenX")]
    pub fn get_blob_screen_x(&self) -> f32 {
        self.game.blob_screen_pos_cached[0]
    }

    #[wasm_bindgen(js_name = "getBlobScreenY")]
    pub fn get_blob_screen_y(&self) -> f32 {
        self.game.blob_screen_pos_cached[1]
    }
}
3. Renderer.astro: Rolling DRS & Zero-Allocation Loop

This replaces your renderLoop to fix the stutters and mobile battery drain.

TypeScript
// Renderer.astro

// ... existing imports
// --- Performance Variables ---
const fpsHistory: number[] = [];
const HISTORY_SIZE = 10; // Smooth over ~10 frames
const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;

function renderLoop(timestamp: number) {
    if (!canvas || !canvas.isConnected) { cleanup(); return; }
    if (!isVisible || document.hidden) { rafId = requestAnimationFrame(renderLoop); return; }

    const delta = timestamp - lastFrameTime;
    lastFrameTime = timestamp;

    // --- SOTA FPS Rolling Average ---
    frameTimer += delta;
    frameCount++;
    
    // Check every 200ms
    if (frameTimer >= 200) { 
        const currentFps = (frameCount * 1000) / frameTimer;
        fpsHistory.push(currentFps);
        if (fpsHistory.length > HISTORY_SIZE) fpsHistory.shift();
        
        const avgFps = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length;
        
        [cite_start]// Mobile Cap Logic: 1.5x is the sweet spot for battery/quality balance [cite: 1]
        const hardCapDpr = isMobile ? 1.5 : 2.0;

        if (drsCooldown > 0) {
            drsCooldown--;
        } else {
            // Smoother Downgrade
            if (avgFps < 45 && targetDpr > 0.5) {
                targetDpr = Math.max(0.5, targetDpr - 0.1); 
                triggerResize();
                drsCooldown = 5; 
            } 
            // Conservative Upgrade
            else if (avgFps > 58 && targetDpr < hardCapDpr) {
                targetDpr = Math.min(hardCapDpr, targetDpr + 0.1);
                triggerResize();
                drsCooldown = 5;
            }
        }
        frameTimer = 0;
        frameCount = 0;
    }

    if (state) {
        try {
            // --- Zero-Allocation UI Repulsion ---
            [cite_start]// Replaces getBlobScreenPosition() array allocation [cite: 2]
            if (isBlobActive.get()) {
                // Check if the getter exists (backward compatibility)
                if (state.getBlobScreenX) {
                    const x = state.getBlobScreenX(); 
                    const y = state.getBlobScreenY();

                    // Rust initializes to -1.0, so check bounds
                    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
                        applyUIRepulsion(x, y);
                    } else {
                        resetUIRepulsion();
                    }
                }
            } else {
                resetUIRepulsion();
            }

            state.render(timestamp * 0.001);
        } catch (e) {
            console.error("Render Crash:", e);
            cleanup();
            return;
        }
    }
    rafId = requestAnimationFrame(renderLoop);
}

// Updated triggerResize to enforce mobile caps
function triggerResize() {
     if (!container || !canvas) return;

     const pixelRatio = window.devicePixelRatio || 1;
     // Enforce Hard Cap
     const hardCap = isMobile ? 1.5 : 2.0;
     const maxAllowed = Math.min(pixelRatio, hardCap);
     
     // Low Power Mode Override
     const effectiveCap = isFallbackMode ? 1.0 : maxAllowed;

     currentDpr = Math.min(targetDpr, effectiveCap);
     
     // ... [rest of your resize logic] ...
}