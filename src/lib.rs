mod uniforms;
mod resources;
mod renderer;
mod state;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use js_sys::Array;
use std::panic;
use renderer::Renderer;
use state::GameState;
use resources::load_model_from_bytes;

// --- CONSTANTS ---
// OPTIMIZATION: 4x MSAA for SOTA quality
const SAMPLE_COUNT: u32 = 4;

#[wasm_bindgen]
pub struct State {
    renderer: Renderer,
    game: GameState,
    mouse_x: f32,
    mouse_y: f32,
    screen_width: f32,
    screen_height: f32,
}

#[wasm_bindgen]
impl State {
    #[wasm_bindgen(js_name = "setTheme")]
    pub fn set_theme(&mut self, is_dark: bool) {
        self.game.is_dark_theme = is_dark;
        
        if is_dark {
            self.game.base_light_color = [0.96, 0.98, 1.0, 1.0];
            self.game.base_sky_color = [0.08, 0.1, 0.15, 1.0];
        } else {
            self.game.base_light_color = [1.0, 0.96, 0.92, 1.0];
            self.game.base_sky_color = [0.92, 0.94, 0.98, 1.0];
        }
        
        web_sys::console::log_1(&format!("Theme updated to: {}", if is_dark { "Dark" } else { "Light" }).into());
        self.game.update_theme_lighting();
    }
    
    #[wasm_bindgen(js_name = "setEnvironmentLight")]
    pub fn set_environment_light(&mut self, sky_r: f32, sky_g: f32, sky_b: f32, light_r: f32, light_g: f32, light_b: f32) {
        self.game.base_sky_color = [sky_r, sky_g, sky_b, 1.0];
        self.game.base_light_color = [light_r, light_g, light_b, 1.0];
        self.game.update_theme_lighting();
    }

    #[wasm_bindgen(js_name = "setCursorLightActive")]
    pub fn set_cursor_light_active(&mut self, active: bool) {
        self.game.cursor_light_active = active;
        
        let blob_active = self.game.blob_exists && self.game.blob_light_enabled;
        let cursor_active = self.game.cursor_light_active;
        
        if blob_active && cursor_active {
            self.game.light_pos_3d = self.game.blob_light_pos_3d * 0.6 + self.game.cursor_light_pos_3d * 0.4;
        } else if blob_active {
            self.game.light_pos_3d = self.game.blob_light_pos_3d;
        } else if cursor_active {
            self.game.light_pos_3d = self.game.cursor_light_pos_3d;
        } else {
            self.game.light_pos_3d = vec3(0.0, 5.0, 0.0);
        }
        self.game.update_theme_lighting();
    }
    
    #[wasm_bindgen(js_name = "loadModelFromBytes")]
    pub fn load_model_from_bytes(&mut self, bytes: &[u8]) {
        match load_model_from_bytes(
            &self.renderer.device, 
            &self.renderer.queue, 
            bytes, 
            &self.renderer.material_layout,
            &self.renderer.mipmap_pipeline_linear,
            &self.renderer.mipmap_pipeline_srgb,
            &self.renderer.mipmap_bind_group_layout,
            self.game.tx.clone()
        ) {
            Ok(model) => {
                self.game.model_center = model.center;
                self.game.model_extent = model.extent;
                self.game.model = Some(model);
                web_sys::console::log_1(&format!("Model loaded with textures! Center: {:?}", self.game.model_center).into());
            },
            Err(e) => {
                web_sys::console::error_1(&format!("Failed to parse GLB: {}", e).into());
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
        self.game.resize(width, height);
        self.game.update_camera_uniforms();
    }

    pub fn update_camera(&mut self, dx: f32, dy: f32, zoom: f32) {
        let sensitivity = 0.005;
        self.game.camera_azimuth -= dx * sensitivity;
        self.game.camera_polar = (self.game.camera_polar - dy * sensitivity).clamp(0.01, std::f32::consts::PI - 0.01);
        self.game.camera_radius = (self.game.camera_radius + zoom * 0.002 * self.game.camera_radius).clamp(0.5, 50.0);
        self.game.update_camera_uniforms();
    }

    pub fn update(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) {
        self.mouse_x = mouse_x;
        self.mouse_y = mouse_y;
        self.screen_width = screen_width;
        self.screen_height = screen_height;
    }

    #[wasm_bindgen(js_name = "updateBlobY")]
    pub fn update_blob_y(&mut self, delta_y: f32) {
        if self.game.blob_exists && self.game.blob_dragging {
            let y_change = -delta_y * 0.005;
            let new_y = (self.game.blob_target_position.y + y_change).clamp(
                self.game.model_center.y - 2.0,
                self.game.model_center.y + 10.0
            );
            self.game.blob_target_position.y = new_y;
        }
    }
    
    #[wasm_bindgen(js_name = "spawnBlob")]
    pub fn spawn_blob(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) {
        if !self.game.blob_exists {
            let mouse_norm_x = (mouse_x / screen_width) * 2.0 - 1.0;
            let mouse_norm_y = 1.0 - (mouse_y / screen_height) * 2.0;

            let x = self.game.camera_radius * self.game.camera_polar.sin() * self.game.camera_azimuth.sin();
            let y = self.game.camera_radius * self.game.camera_polar.cos();
            let z = self.game.camera_radius * self.game.camera_polar.sin() * self.game.camera_azimuth.cos();
            let cam_pos = vec3(x, y, z) + self.game.camera_target;
            
            let view = Mat4::look_at_rh(cam_pos, self.game.camera_target, Vec3::Y);
            let proj = Mat4::perspective_rh(45.0_f32.to_radians(), screen_width / screen_height, 0.1, 100.0);
            let inv_view_proj = (proj * view).inverse();

            let near_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 0.0, 1.0);
            let far_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 1.0, 1.0);
            let near_world = vec3(near_point.x, near_point.y, near_point.z) / near_point.w;
            let far_world = vec3(far_point.x, far_point.y, far_point.z) / far_point.w;
            let ray_dir = (far_world - near_world).normalize();
            
            let spawn_distance = self.game.model_extent * 1.5;
            let spawn_position = near_world + ray_dir * spawn_distance;

            self.game.blob_position = spawn_position;
            self.game.blob_prev_position = spawn_position;
            self.game.blob_target_position = spawn_position;
            self.game.blob_light_enabled = true;
            self.game.blob_dragging = false;
            self.game.blob_light_pos_3d = self.game.blob_position;

            self.game.scene_uniform.blob.position = [spawn_position.x, spawn_position.y, spawn_position.z, 0.0];
            self.game.blob_exists = true;

            self.game.update_theme_lighting();
        }
    }
    
    #[wasm_bindgen(js_name = "despawnBlob")]
    pub fn despawn_blob(&mut self) {
        self.game.blob_exists = false;
        self.game.blob_dragging = false;
        self.game.update_theme_lighting();
    }
    
    #[wasm_bindgen(js_name = "toggleBlobLight")]
    pub fn toggle_blob_light(&mut self) {
        if self.game.blob_exists {
            self.game.blob_light_enabled = !self.game.blob_light_enabled;
            if self.game.blob_light_enabled {
                self.game.blob_light_pos_3d = self.game.blob_position;
            }
            self.game.update_theme_lighting();
        }
    }
    
    #[wasm_bindgen(js_name = "isHoveringBlob")]
    pub fn is_hovering_blob(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.game.blob_exists {
            return false;
        }
        
        let world_pos = self.game.blob_position;
        let screen_pos = self.project_3d_to_screen(world_pos, screen_width, screen_height);
        let mouse_norm_x = mouse_x / screen_width;
        let mouse_norm_y = mouse_y / screen_height;
        
        let dist = ((screen_pos[0] - mouse_norm_x).powi(2) + (screen_pos[1] - mouse_norm_y).powi(2)).sqrt();
        
        dist < 0.15 && screen_pos[0] > 0.0 && screen_pos[0] < 1.0 && screen_pos[1] > 0.0 && screen_pos[1] < 1.0
    }
    
    #[wasm_bindgen(js_name = "startDragBlob")]
    pub fn start_drag_blob(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.game.blob_exists {
            return false;
        }
        
        if self.is_hovering_blob(mouse_x, mouse_y, screen_width, screen_height) {
            self.game.blob_dragging = true;
            self.game.blob_target_position = self.game.blob_position;
            
            let x = self.game.camera_radius * self.game.camera_polar.sin() * self.game.camera_azimuth.sin();
            let y = self.game.camera_radius * self.game.camera_polar.cos();
            let z = self.game.camera_radius * self.game.camera_polar.sin() * self.game.camera_azimuth.cos();
            let cam_pos = vec3(x, y, z) + self.game.camera_target;
            let view = Mat4::look_at_rh(cam_pos, self.game.camera_target, Vec3::Y);
            let proj = Mat4::perspective_rh(45.0_f32.to_radians(), screen_width / screen_height, 0.1, 100.0);
            let blob_clip = proj * view * vec4(self.game.blob_position.x, self.game.blob_position.y, self.game.blob_position.z, 1.0);
            let blob_ndc_z = blob_clip.z / blob_clip.w;
            self.game.blob_drag_depth = blob_ndc_z.clamp(0.0, 1.0);
            
            return true;
        }
        
        false
    }
    
    #[wasm_bindgen(js_name = "stopDragBlob")]
    pub fn stop_drag_blob(&mut self) {
        self.game.blob_dragging = false;
        self.game.blob_drag_depth = 0.0;
    }
    
    #[wasm_bindgen(js_name = "checkBlobClick")]
    pub fn check_blob_click(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.game.blob_exists {
            return false;
        }
        
        if self.is_hovering_blob(mouse_x, mouse_y, screen_width, screen_height) {
            if !self.game.blob_dragging {
                self.toggle_blob_light();
            }
            return true;
        }
        
        false
    }

    fn project_3d_to_screen(&self, world_pos: Vec3, screen_width: f32, screen_height: f32) -> [f32; 2] {
        let x = self.game.camera_radius * self.game.camera_polar.sin() * self.game.camera_azimuth.sin();
        let y = self.game.camera_radius * self.game.camera_polar.cos();
        let z = self.game.camera_radius * self.game.camera_polar.sin() * self.game.camera_azimuth.cos();
        let cam_pos = vec3(x, y, z) + self.game.camera_target;
        
        let view = Mat4::look_at_rh(cam_pos, self.game.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), screen_width / screen_height, 0.1, 100.0);
        let view_proj = proj * view;
        
        let world = vec4(world_pos.x, world_pos.y, world_pos.z, 1.0);
        let clip = view_proj * world;
        
        let w = clip.w;
        if w.abs() < 0.0001 {
            return [0.5, 0.5];
        }
        
        let ndc_x = clip.x / w;
        let ndc_y = clip.y / w;
        
        let screen_x = (ndc_x + 1.0) * 0.5;
        let screen_y = 1.0 - (ndc_y + 1.0) * 0.5;
        
        [screen_x, screen_y]
    }
    
    #[wasm_bindgen(js_name = "getLightScreenPos")]
    pub fn get_light_screen_pos(&self) -> Array {
        let width = self.renderer.config.width as f32;
        let height = self.renderer.config.height as f32;
        let screen_pos = self.project_3d_to_screen(self.game.light_pos_3d, width, height);
        
        let result = Array::new_with_length(2);
        result.set(0, JsValue::from_f64(screen_pos[0] as f64));
        result.set(1, JsValue::from_f64(screen_pos[1] as f64));
        result
    }

    #[wasm_bindgen(js_name = "getBlobScreenPosition")]
    pub fn get_blob_screen_position(&self) -> Array {
        if !self.game.blob_exists {
            return Array::new();
        }
        
        let width = self.renderer.config.width as f32;
        let height = self.renderer.config.height as f32;
        let screen_pos = self.project_3d_to_screen(self.game.blob_position, width, height);
        
        let result = Array::new_with_length(2);
        result.set(0, JsValue::from_f64(screen_pos[0] as f64));
        result.set(1, JsValue::from_f64(screen_pos[1] as f64));
        result
    }

    pub fn render(&mut self, time: f32) {
        self.game.update(time as f64, self.mouse_x, self.mouse_y, self.screen_width, self.screen_height, &self.renderer.device, &self.renderer.material_layout);
        self.renderer.render(&self.game.scene_uniform, self.game.model.as_ref(), Some(&self.game.blob_mesh), self.game.blob_exists);
    }
}

#[wasm_bindgen(js_name = "startRenderer")]
pub async fn start_renderer(canvas: HtmlCanvasElement, is_mobile: bool) -> Result<State, JsValue> {
    #[cfg(feature = "console_error_panic_hook")]
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    web_sys::console::log_1(&format!("Initializing Advanced WGPU Renderer... Mobile: {}", is_mobile).into());

    let renderer = Renderer::new(canvas, is_mobile, SAMPLE_COUNT).await?;
    let game = GameState::new(&renderer.device, &renderer.queue, &renderer.material_layout);
    
    Ok(State { 
        renderer, 
        game,
        mouse_x: 0.0,
        mouse_y: 0.0,
        screen_width: 1.0,
        screen_height: 1.0,
    })
}
