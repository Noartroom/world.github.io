use wgpu::util::DeviceExt;
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use std::rc::Rc;
use web_sys::HtmlCanvasElement;
use wasm_bindgen::JsValue;
use crate::uniforms::{SceneUniform, ModelVertex};
use crate::resources::{Mesh, Model};

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: Rc<wgpu::Device>,
    pub queue: Rc<wgpu::Queue>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub msaa_texture: Option<wgpu::Texture>,
    pub msaa_view: Option<wgpu::TextureView>,
    pub sample_count: u32,
    pub opaque_pipeline: wgpu::RenderPipeline,
    pub transparent_pipeline: wgpu::RenderPipeline,
    pub blob_pipeline: wgpu::RenderPipeline,
    pub mipmap_pipeline_linear: Rc<wgpu::RenderPipeline>,
    pub mipmap_pipeline_srgb: Rc<wgpu::RenderPipeline>,
    pub mipmap_bind_group_layout: Rc<wgpu::BindGroupLayout>,
    pub material_layout: wgpu::BindGroupLayout,
    pub scene_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub async fn new(canvas: HtmlCanvasElement, is_mobile: bool, sample_count: u32) -> Result<Self, JsValue> {
        // Force GL-only on mobile/iOS to avoid WebGPU adapter failures; keep WebGPU enabled on desktop.
        let backends = if is_mobile {
            wgpu::Backends::GL
        } else {
            wgpu::Backends::GL | wgpu::Backends::BROWSER_WEBGPU
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        #[cfg(target_arch = "wasm32")]
        let target = wgpu::SurfaceTarget::Canvas(canvas.clone());
        
        #[cfg(not(target_arch = "wasm32"))]
        let target: wgpu::SurfaceTarget = unimplemented!("Not supported on non-WASM targets");

        let surface = instance.create_surface(target)
            .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {}", e)))?;
        
        // Adapter selection matrix to survive blocked WebGPU and missing surface support
        let adapter = async {
            // 1) HighPerformance + surface
            if let Some(a) = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }).await {
                return Some(a);
            }
            // 2) HighPerformance + surface + fallback (can map to GL)
            if let Some(a) = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: true,
            }).await {
                return Some(a);
            }
            // 3) LowPower + surface (some mobile GPUs only expose this)
            if let Some(a) = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }).await {
                return Some(a);
            }
            // 4) LowPower + surface + fallback
            if let Some(a) = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: true,
            }).await {
                return Some(a);
            }
            // 5) HighPerformance without surface (last resort; may still allow GL)
            if let Some(a) = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: true,
            }).await {
                return Some(a);
            }
            // 6) LowPower without surface
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }).await
        }
        .await
        .ok_or_else(|| JsValue::from_str("No adapter (including fallback)"))?;
        
        let mut required_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_defaults() // allow GL1/GL2 minimal limits
        } else {
            wgpu::Limits::downlevel_webgl2_defaults()
        };
        let adapter_limits = adapter.limits();
        required_limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
        required_limits.max_compute_workgroups_per_dimension = 0;
        required_limits.max_compute_invocations_per_workgroup = 0;
        required_limits.max_compute_workgroup_storage_size = 0;
        required_limits.max_compute_workgroup_size_x = 0;
        required_limits.max_compute_workgroup_size_y = 0;
        required_limits.max_compute_workgroup_size_z = 0;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits,
                label: None,
            },
            None,
        ).await.map_err(|e| JsValue::from_str(&e.to_string()))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| surface_caps.formats.iter().copied().find(|f| matches!(f, wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm)).unwrap_or(wgpu::TextureFormat::Rgba8Unorm));

        let present_mode = surface_caps.present_modes.iter()
            .copied()
            .find(|&mode| mode == wgpu::PresentMode::Mailbox)
            .unwrap_or(wgpu::PresentMode::Fifo);

        let alpha_mode = surface_caps.alpha_modes.iter()
            .copied()
            .find(|&mode| mode == wgpu::CompositeAlphaMode::PreMultiplied)
            .or_else(|| surface_caps.alpha_modes.iter().copied().find(|&mode| mode == wgpu::CompositeAlphaMode::PostMultiplied))
            .unwrap_or(surface_caps.alpha_modes[0]);

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        
        let info = adapter.get_info();
        let actual_sample_count = if info.backend == wgpu::Backend::Gl {
            web_sys::console::warn_1(&"WebGL backend detected, disabling MSAA to prevent panic.".into());
            1
        } else if is_mobile {
            2
        } else {
            sample_count
        };
        
        let (depth_texture, depth_view) = Self::create_depth_texture(&device, &config, actual_sample_count);
        let (msaa_texture, msaa_view) = Self::create_msaa_texture(&device, &config, actual_sample_count)
            .map(|(t, v)| (Some(t), Some(v)))
            .unwrap_or((None, None));

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // Scene Uniform
        let scene_uniform = SceneUniform {
            camera: crate::uniforms::CameraUniform { 
                view_proj: Mat4::IDENTITY.to_cols_array_2d(), 
                inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(), 
                camera_pos: [0.0; 4] 
            },
            light: crate::uniforms::LightUniform {
                position: [0.0; 4],
                color: [0.0; 4],
                sky_color: [0.0; 4],
            },
            blob: crate::uniforms::BlobUniform { position: [0.0; 4], color: [0.0; 4] },
            time: 0.0,
            _padding: [0; 3],
        };
        
        let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Buffer"),
            contents: bytemuck::cast_slice(&[scene_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Scene Layout"),
        });
        
        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &scene_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
            label: Some("Scene Bind Group"),
        });

        // Material Layout
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ], label: Some("Material Layout")
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&scene_layout, &material_layout],
            push_constant_ranges: &[],
        });

        // Pipelines
        let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Opaque Pipeline"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_model", buffers: &[ModelVertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_model", targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() }, 
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState { count: actual_sample_count, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None
        });

        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Transparent Pipeline"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_model", buffers: &[ModelVertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_model", targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState { count: actual_sample_count, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None
        });

        let blob_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blob Pipeline"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_blob", buffers: &[ModelVertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_model", targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState { count: actual_sample_count, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None
        });

        // Mipmap Pipeline
        let mipmap_bind_group_layout = Rc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("Mipmap Bind Group Layout"),
        }));

        let mipmap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mipmap Pipeline Layout"),
            bind_group_layouts: &[&mipmap_bind_group_layout],
            push_constant_ranges: &[],
        });

        let create_mipmap_pipeline = |format: wgpu::TextureFormat, label: &str| -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&mipmap_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_mipmap",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_mipmap",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };

        let mipmap_pipeline_linear = Rc::new(create_mipmap_pipeline(wgpu::TextureFormat::Rgba8Unorm, "Mipmap Pipeline (Linear)"));
        let mipmap_pipeline_srgb = Rc::new(create_mipmap_pipeline(wgpu::TextureFormat::Rgba8UnormSrgb, "Mipmap Pipeline (sRGB)"));

        Ok(Self {
            surface,
            device: Rc::new(device),
            queue: Rc::new(queue),
            config,
            depth_texture,
            depth_view,
            msaa_texture,
            msaa_view,
            sample_count: actual_sample_count,
            opaque_pipeline,
            transparent_pipeline,
            blob_pipeline,
            mipmap_pipeline_linear,
            mipmap_pipeline_srgb,
            mipmap_bind_group_layout,
            material_layout,
            scene_buffer,
            scene_bind_group,
        })
    }

    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, sample_count: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, sample_count: u32) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            
            let (depth_texture, depth_view) = Self::create_depth_texture(&self.device, &self.config, self.sample_count);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
            
            if let Some((msaa_texture, msaa_view)) = Self::create_msaa_texture(&self.device, &self.config, self.sample_count) {
                self.msaa_texture = Some(msaa_texture);
                self.msaa_view = Some(msaa_view);
            } else {
                self.msaa_texture = None;
                self.msaa_view = None;
            }
        }
    }

    fn is_aabb_in_frustum(view_proj: &[[f32; 4]; 4], aabb_min: Vec3, aabb_max: Vec3) -> bool {
        let m = view_proj;
        let planes = [
            vec4(m[3][0] + m[0][0], m[3][1] + m[0][1], m[3][2] + m[0][2], m[3][3] + m[0][3]),
            vec4(m[3][0] - m[0][0], m[3][1] - m[0][1], m[3][2] - m[0][2], m[3][3] - m[0][3]),
            vec4(m[3][0] + m[1][0], m[3][1] + m[1][1], m[3][2] + m[1][2], m[3][3] + m[1][3]),
            vec4(m[3][0] - m[1][0], m[3][1] - m[1][1], m[3][2] - m[1][2], m[3][3] - m[1][3]),
            vec4(m[3][0] + m[2][0], m[3][1] + m[2][1], m[3][2] + m[2][2], m[3][3] + m[2][3]),
            vec4(m[3][0] - m[2][0], m[3][1] - m[2][1], m[3][2] - m[2][2], m[3][3] - m[2][3]),
        ];

        let planes: Vec<Vec4> = planes.iter().map(|p| {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if len > 0.0001 { *p / len } else { *p }
        }).collect();

        for plane in planes {
            let p_vertex = vec3(
                if plane.x >= 0.0 { aabb_max.x } else { aabb_min.x },
                if plane.y >= 0.0 { aabb_max.y } else { aabb_min.y },
                if plane.z >= 0.0 { aabb_max.z } else { aabb_min.z },
            );

            let distance = plane.x * p_vertex.x + plane.y * p_vertex.y + plane.z * p_vertex.z + plane.w;
            if distance < 0.0 {
                return false;
            }
        }
        true
    }

    pub fn render(&mut self, scene_uniform: &SceneUniform, model: Option<&Model>, blob_mesh: Option<&Mesh>, blob_exists: bool) {
        self.queue.write_buffer(&self.scene_buffer, 0, bytemuck::cast_slice(&[*scene_uniform]));

        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => {
                self.resize(self.config.width, self.config.height);
                return;
            }
            Err(_) => return,
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        {
            let (color_view, resolve_target) = if let Some(ref msaa_view) = self.msaa_view {
                (msaa_view, Some(&view))
            } else {
                (&view, None)
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_bind_group(0, &self.scene_bind_group, &[]);

            if let Some(model) = model {
                render_pass.set_pipeline(&self.opaque_pipeline);
                
                for mesh in &model.opaque_meshes {
                    if !Self::is_aabb_in_frustum(&scene_uniform.camera.view_proj, mesh.aabb_min, mesh.aabb_max) {
                        continue;
                    }
                    
                    render_pass.set_bind_group(1, &mesh.material_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }

                // Transparent sorting
                // We need camera position to sort. It's in scene_uniform.camera.camera_pos (vec4)
                let cam_pos = vec3(
                    scene_uniform.camera.camera_pos[0],
                    scene_uniform.camera.camera_pos[1],
                    scene_uniform.camera.camera_pos[2]
                );
                
                let mut transparent_to_draw: Vec<(&Mesh, f32)> = model.transparent_meshes.iter()
                    .filter_map(|mesh| {
                        if !Self::is_aabb_in_frustum(&scene_uniform.camera.view_proj, mesh.aabb_min, mesh.aabb_max) {
                            return None;
                        }
                        let distance = (mesh.center - cam_pos).length_squared();
                        Some((mesh, distance))
                    })
                    .collect();
                
                transparent_to_draw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                render_pass.set_pipeline(&self.transparent_pipeline);
                for (mesh, _) in transparent_to_draw {
                    render_pass.set_bind_group(1, &mesh.material_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }

            if blob_exists {
                if let Some(blob_mesh) = blob_mesh {
                    render_pass.set_pipeline(&self.blob_pipeline);
                    render_pass.set_bind_group(1, &blob_mesh.material_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, blob_mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(blob_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..blob_mesh.num_indices, 0, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
