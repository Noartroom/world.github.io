Goal: Fix the "Hot Loop" (CPU usage) and "GC Killer" (Stutter) simultaneously.

Modify state.rs We hoist the ray calculation out of the physics loop and add the cache field.

Rust
// state.rs
use glam::{vec3, vec4, Mat4, Vec3, Vec4};

// 1. Structure to hold the Ray (replaces re-calculation)
pub struct WorldRay {
    pub origin: Vec3,
    pub direction: Vec3,
}

pub struct GameState {
    // ... existing fields ...
    
    // 2. Cache for Zero-Copy Access (Fixes GC Stutter)
    pub blob_screen_pos_cached: [f32; 2], 
}

impl GameState {
    pub fn new(...) -> Self {
        Self {
            // ... [existing init] ...
            blob_screen_pos_cached: [-1.0; 2], // Init off-screen
        }
    }

    // 3. Optimized Update Loop
    pub fn update(&mut self, time: f64, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32, ...) {
        // [Asset Loading logic...]

        // STEP A: Calculate Ray ONCE (Hoisted out of the loop)
        // 
        let mouse_ray = self.calculate_mouse_ray(mouse_x, mouse_y, screen_width, screen_height);

        // STEP B: Physics Loop (Now lightweight)
        let frame_time = time - self.last_time;
        self.last_time = time;
        self.accumulator += frame_time;
        // Safety cap
        if self.accumulator > 0.25 { self.accumulator = 0.25; }

        const DT: f64 = 1.0 / 60.0;
        while self.accumulator >= DT {
            // Pass the pre-calculated ray
            self.step_physics(&mouse_ray); 
            self.accumulator -= DT;
        }

        // STEP C: Zero-Alloc Screen Position
        if self.blob_exists {
            // Calculate this ONCE per frame, not per JS call
            self.blob_screen_pos_cached = self.project_3d_to_screen(self.blob_position, screen_width, screen_height);
            
            // [Interpolation logic...]
        }
    }

    // Helper: Moves the heavy matrix math here
    fn calculate_mouse_ray(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> WorldRay {
        let mouse_norm_x = (mouse_x / screen_width) * 2.0 - 1.0;
        let mouse_norm_y = 1.0 - (mouse_y / screen_height) * 2.0;
        
        // [Insert your existing Matrix Inversion / Unproject code here]
        // ...
        // let near_world = ...
        // let far_world = ...

        WorldRay {
            origin: near_world,
            direction: (far_world - near_world).normalize(),
        }
    }

    // Update step_physics to use the ray
    fn step_physics(&mut self, ray: &WorldRay) {
        // Cursor Light
        if self.cursor_light_active {
            self.cursor_light_pos_3d = ray.origin + ray.direction * 3.0;
        }
        // ... [Rest of physics using ray.origin/direction instead of recalculating]
    }
}
Modify lib.rs Expose the scalar getters for the bridge.

Rust
// lib.rs
#[wasm_bindgen]
impl State {
    // ... [existing methods]

    // NEW: Zero-Copy Getters
    #[wasm_bindgen(js_name = "getBlobScreenX")]
    pub fn get_blob_screen_x(&self) -> f32 {
        self.game.blob_screen_pos_cached[0]
    }

    #[wasm_bindgen(js_name = "getBlobScreenY")]
    pub fn get_blob_screen_y(&self) -> f32 {
        self.game.blob_screen_pos_cached[1]
    }
}