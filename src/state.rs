use glam::{vec3, vec4, Mat4, Vec3};
use std::rc::Rc;
use std::collections::HashMap;
use flume::{Sender, Receiver};
use wgpu;
use crate::uniforms::{SceneUniform, CameraUniform, LightUniform, BlobUniform};
use crate::resources::{Model, Mesh, AssetMessage, create_sphere_mesh};

pub struct WorldRay {
    pub origin: Vec3,
    pub direction: Vec3,
}

pub struct GameState {
    // Game State
    pub scene_uniform: SceneUniform,
    
    // Camera
    pub camera_target: Vec3,
    pub camera_radius: f32,
    pub camera_azimuth: f32,
    pub camera_polar: f32,
    
    // Light / Theme
    pub is_dark_theme: bool,
    pub base_light_color: [f32; 4],
    pub base_sky_color: [f32; 4],
    pub light_pos_3d: Vec3,
    pub cursor_light_pos_3d: Vec3,
    pub cursor_light_active: bool,
    
    // Blob
    pub blob_exists: bool,
    pub blob_position: Vec3,
    pub blob_prev_position: Vec3, // For interpolation
    pub blob_target_position: Vec3,
    pub blob_light_enabled: bool,
    pub blob_light_pos_3d: Vec3,
    pub blob_dragging: bool,
    pub blob_drag_depth: f32,
    
    // Model
    pub model: Option<Model>,
    pub model_center: Vec3,
    pub model_extent: f32,
    
    // Blob Mesh (created once)
    pub blob_mesh: Mesh,
    pub blob_screen_pos_cached: [f32; 2],
    
    // Asset Loading
    pub texture_cache: HashMap<usize, Rc<wgpu::TextureView>>,
    pub tx: Sender<AssetMessage>,
    pub rx: Receiver<AssetMessage>,

    // Physics
    pub last_time: f64,
    pub accumulator: f64,
    pub aspect_ratio: f32,
}

impl GameState {
    pub fn new(
        device: &wgpu::Device, 
        queue: &wgpu::Queue, 
        material_layout: &wgpu::BindGroupLayout
    ) -> Self {
        let (tx, rx) = flume::unbounded();

        // Initial colors for Dark Theme (default)
        let base_light = [0.96, 0.98, 1.0, 1.0];
        let base_sky = [0.08, 0.1, 0.15, 1.0];

        let blob_mesh = create_sphere_mesh(device, queue, 0.5, 64, material_layout);

        let scene_uniform = SceneUniform {
            camera: CameraUniform { view_proj: Mat4::IDENTITY.to_cols_array_2d(), inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(), camera_pos: [0.0; 4] },
            light: LightUniform {
                position: [5.0, 5.0, 5.0, 1.0],
                color: base_light,
                sky_color: base_sky,
            },
            blob: BlobUniform { position: [0.0; 4], color: [1.0; 4] },
            time: 0.0,
            _padding: [0; 3],
        };

        Self {
            scene_uniform,
            camera_target: Vec3::ZERO,
            camera_radius: 10.0,
            camera_azimuth: 0.0,
            camera_polar: 1.57,
            is_dark_theme: true,
            base_light_color: base_light,
            base_sky_color: base_sky,
            light_pos_3d: vec3(2.0, 2.0, 2.0),
            cursor_light_pos_3d: vec3(2.0, 2.0, 2.0),
            cursor_light_active: false,
            blob_exists: false,
            blob_position: vec3(0.0, 5.0, 0.0),
            blob_prev_position: vec3(0.0, 5.0, 0.0),
            blob_target_position: vec3(0.0, 5.0, 0.0),
            blob_light_enabled: false,
            blob_light_pos_3d: Vec3::ZERO,
            blob_dragging: false,
            blob_drag_depth: 0.0,
            model: None,
            model_center: Vec3::ZERO,
            model_extent: 2.0,
            blob_mesh,
            blob_screen_pos_cached: [-1.0; 2],
            texture_cache: HashMap::new(),
            tx,
            rx,
            last_time: 0.0,
            accumulator: 0.0,
            aspect_ratio: 1.0,
        }
    }

    fn calculate_mouse_ray(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> WorldRay {
        let safe_width = screen_width.max(1.0);
        let safe_height = screen_height.max(1.0);

        let mouse_norm_x = (mouse_x / safe_width) * 2.0 - 1.0;
        let mouse_norm_y = 1.0 - (mouse_y / safe_height) * 2.0;

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

    pub fn project_3d_to_screen(&self, world_pos: Vec3, screen_width: f32, screen_height: f32) -> [f32; 2] {
        let safe_width = screen_width.max(1.0);
        let safe_height = screen_height.max(1.0);

        let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
        let y = self.camera_radius * self.camera_polar.cos();
        let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
        let cam_pos = vec3(x, y, z) + self.camera_target;
        
        let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), safe_width / safe_height, 0.1, 100.0);
        let view_proj = proj * view;
        
        let world = vec4(world_pos.x, world_pos.y, world_pos.z, 1.0);
        let clip = view_proj * world;
        
        let w = clip.w;
        if w.abs() < 0.0001 {
            return [-1.0, -1.0];
        }
        
        let ndc_x = clip.x / w;
        let ndc_y = clip.y / w;
        
        let screen_x = (ndc_x + 1.0) * 0.5;
        let screen_y = 1.0 - (ndc_y + 1.0) * 0.5;
        
        [screen_x, screen_y]
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect_ratio = width as f32 / height as f32;
        }
    }

    pub fn update_camera_uniforms(&mut self) {
        let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
        let y = self.camera_radius * self.camera_polar.cos();
        let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
        let pos = vec3(x, y, z) + self.camera_target;
        
        let view = Mat4::look_at_rh(pos, self.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), self.aspect_ratio, 0.1, 100.0);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        self.scene_uniform.camera = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, 1.0],
        };
    }

    pub fn update_theme_lighting(&mut self) {
        let light_pos = self.light_pos_3d;
        
        let blob_active = self.blob_exists && self.blob_light_enabled;
        let cursor_active = self.cursor_light_active;
        
        if !blob_active && !cursor_active {
            self.scene_uniform.light = LightUniform {
                position: [light_pos.x, light_pos.y, light_pos.z, 1.0],
                color: [0.0, 0.0, 0.0, 0.0],
                sky_color: [0.0, 0.0, 0.0, 1.0],
            };
            return;
        }

        let base_intensity_boost = 2.0;
        let blob_additional_boost = if blob_active { 0.5 } else { 0.0 };
        let cursor_additional_boost = if cursor_active && !blob_active { 1.1 } else { 0.0 };
        let intensity_boost = base_intensity_boost + blob_additional_boost + cursor_additional_boost;
        
        let base_color = self.base_light_color;
        let max_intensity = if self.is_dark_theme { 3.0 } else { 2.5 };
        let ambient_mult = if blob_active { 1.0 } else { 0.05 };

        self.scene_uniform.light = LightUniform {
            position: [light_pos.x, light_pos.y, light_pos.z, 1.0],
            color: [
                (base_color[0] * intensity_boost).min(max_intensity),
                (base_color[1] * intensity_boost).min(max_intensity),
                (base_color[2] * intensity_boost).min(max_intensity),
                1.0
            ],
            sky_color: [self.base_sky_color[0] * ambient_mult, self.base_sky_color[1] * ambient_mult, self.base_sky_color[2] * ambient_mult, 1.0],
        };
    }

    // Fixed Timestep Physics Update
    pub fn update(&mut self, time: f64, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32, device: &wgpu::Device, material_layout: &wgpu::BindGroupLayout) {
        // Handle assets first
        while let Ok(msg) = self.rx.try_recv() {
            if let Some(model) = &mut self.model {
                let AssetMessage::TextureLoaded { image_index, texture_type, texture } = msg;
                let view = Rc::new(texture.view);
                self.texture_cache.insert(image_index, view.clone());
                
                let process_mesh = |mesh: &mut Mesh| {
                     let mut update = false;
                    if texture_type == 0 && mesh.diffuse_index == Some(image_index) { mesh.diffuse_view = view.clone(); update = true; }
                    if texture_type == 1 && mesh.normal_index == Some(image_index) { mesh.normal_view = view.clone(); update = true; }
                    if texture_type == 2 && mesh.mr_index == Some(image_index) { mesh.mr_view = view.clone(); update = true; }
                    
                    if update {
                         mesh.material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: material_layout,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&mesh.diffuse_view) },
                                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&mesh.sampler) },
                                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&mesh.normal_view) },
                                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&mesh.sampler) },
                                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&mesh.mr_view) },
                                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&mesh.sampler) },
                            ],
                            label: None,
                        });
                    }
                };
                
                for mesh in &mut model.opaque_meshes { process_mesh(mesh); }
                for mesh in &mut model.transparent_meshes { process_mesh(mesh); }
            }
        }
    
        // Physics Loop
        let frame_time = time - self.last_time;
        self.last_time = time;
        self.accumulator += frame_time;
        
        // Clamp accumulator to prevent death spiral
        if self.accumulator > 0.25 {
            self.accumulator = 0.25;
        }

        const DT: f64 = 1.0 / 60.0;
        
        let mouse_ray = self.calculate_mouse_ray(mouse_x, mouse_y, screen_width, screen_height);

        while self.accumulator >= DT {
            self.step_physics(&mouse_ray);
            self.accumulator -= DT;
        }
        
        // Interpolation
        let alpha = (self.accumulator / DT) as f32;
        if self.blob_exists {
            let interpolated_pos = self.blob_prev_position.lerp(self.blob_position, alpha);
             self.scene_uniform.blob = BlobUniform {
                position: [interpolated_pos.x, interpolated_pos.y, interpolated_pos.z, 0.0],
                color: self.base_light_color,
            };
            self.blob_light_pos_3d = interpolated_pos;
        }

        if self.blob_exists {
            let screen_pos = self.project_3d_to_screen(self.blob_position, screen_width, screen_height);
            self.blob_screen_pos_cached = screen_pos;
        } else {
            self.blob_screen_pos_cached = [-1.0; 2];
        }

        // Update other uniforms
        self.update_camera_uniforms();
        
        // Lighting logic
         if self.blob_exists && self.blob_light_enabled {
            // Actually scene_uniform.blob.position is [f32; 4], so take xyz
            let p = self.scene_uniform.blob.position;
            self.blob_light_pos_3d = vec3(p[0], p[1], p[2]);
        }
        
        let blob_active = self.blob_exists && self.blob_light_enabled;
        let cursor_active = self.cursor_light_active;
        
        if blob_active {
            self.light_pos_3d = self.blob_light_pos_3d;
        } else if cursor_active {
            self.light_pos_3d = self.cursor_light_pos_3d;
        } else {
            self.light_pos_3d = vec3(0.0, 5.0, 0.0);
        }
        self.update_theme_lighting();
    }
    
    fn step_physics(&mut self, ray: &WorldRay) {
        // Store previous state for interpolation
        self.blob_prev_position = self.blob_position;

        // Cursor light
        if self.cursor_light_active {
            let light_distance = 3.0;
            self.cursor_light_pos_3d = ray.origin + ray.direction * light_distance;
        }

        if self.blob_dragging && self.blob_exists {
            let distance_to_center = (self.model_center - ray.origin).length();
            let target_pos = ray.origin + ray.direction * distance_to_center;

            let to_model = target_pos - self.model_center;
            let base_radius = (self.model_extent * 0.5).max(3.0);
            let min_radius = base_radius * 0.5;
            let distance = to_model.length();
            
            let final_position = if distance < min_radius {
                if distance > 0.001 {
                    let direction = to_model / distance;
                    self.model_center + direction * min_radius
                } else {
                    self.model_center + vec3(min_radius, 0.0, 0.0)
                }
            } else {
                target_pos
            };
            
            self.blob_target_position = final_position;
        }
        
        if self.blob_exists {
            let smoothing_factor = if self.blob_dragging { 0.8 } else { 0.15 };
            let diff = self.blob_target_position - self.blob_position;
            self.blob_position = self.blob_position + diff * smoothing_factor;
        }
    }
}
