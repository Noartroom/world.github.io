use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, ImageBitmap, Blob, BlobPropertyBag, CanvasRenderingContext2d};
use wgpu::util::DeviceExt;
use glam::{vec2, vec3, vec4, Mat4, Vec3, Vec4, Quat};
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use flume::{Sender, Receiver};
use wasm_bindgen_futures::JsFuture;
use js_sys::{Uint8Array, Array};
use std::panic;

// --- CONSTANTS ---
// OPTIMIZATION: 4x MSAA for SOTA quality
const SAMPLE_COUNT: u32 = 4;

// --- Uniforms ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4], // Added for Skybox/Advanced effects
    camera_pos: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 4],
    color: [f32; 4],
    sky_color: [f32; 4],
}

// NEW: Uniform to move the blob cheaply on GPU
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlobUniform {
    position: [f32; 4],
    color: [f32; 4],
}

// NEW: Consolidated Scene Uniform to reduce bind groups
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    camera: CameraUniform,
    light: LightUniform,
    blob: BlobUniform,
    time: f32,             // 4 bytes
    _padding: [u32; 3],    // 12 bytes -> Total 16 bytes (Aligns correctly)
}

// --- Vertex Data ---
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coord: [f32; 2],
    tangent: [f32; 4],
}

impl ModelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x4
    ];
    
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// --- Texture & Async Loading System ---

enum AssetMessage {
    TextureLoaded { 
        image_index: usize, 
        texture_type: u32, 
        texture: Texture 
    }
}

struct Texture {
    view: wgpu::TextureView,
}

impl Texture {
    // Calculate number of mip levels for a texture
    fn calculate_mip_levels(width: u32, height: u32) -> u32 {
        if width == 0 || height == 0 {
            return 1;
        }
        let max_dim = width.max(height) as f32;
        (max_dim.log2().floor() as u32).max(1)
    }

    pub fn from_bitmap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bitmap: ImageBitmap,
        is_srgb: bool,
        mipmap_pipeline: Option<&wgpu::RenderPipeline>,
        mipmap_bind_group_layout: Option<&wgpu::BindGroupLayout>,
    ) -> Self {
        let width = bitmap.width();
        let height = bitmap.height();
        let mip_level_count = Self::calculate_mip_levels(width, height);
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        // Mipmap generation requires non-sRGB view format for storage binding if using COMPUTE,
        // but we are using RENDER pipeline.
        // The issue is likely that we are rendering TO the texture view.
        // If format is Rgba8UnormSrgb, the view will be sRGB.
        // If the pipeline target format doesn't match, we get validation error.
        // We pass 'format' to generate_mipmaps which sets the pipeline target format.
        let format = if is_srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bitmap Texture"),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Use conditional compilation to attempt zero-copy upload on wasm32,
        // with a robust fallback for other platforms or failures.
        let mut uploaded = false;

        #[cfg(target_arch = "wasm32")]
        {
            // The `copy_external_image_to_texture` feature might not be available
            // on all wasm targets, so we still treat it as a potential failure point.
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                queue.copy_external_image_to_texture(
                    &wgpu::ImageCopyExternalImage {
                        source: wgpu::ExternalImageSource::ImageBitmap(bitmap.clone()),
                        origin: wgpu::Origin2d::ZERO,
                        flip_y: false,
                    },
                    wgpu::ImageCopyTextureTagged {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                        color_space: wgpu::PredefinedColorSpace::Srgb,
                        premultiplied_alpha: false,
                    },
                    size,
                );
            }));

            if result.is_ok() {
                uploaded = true;
            } else {
                web_sys::console::warn_1(&"copy_external_image_to_texture failed, falling back to canvas upload".into());
            }
        }

        if !uploaded {
            // Fallback for non-wasm platforms or if the zero-copy method fails.
            // This path is now taken if the compilation target is not wasm32,
            // or if the `copy_external_image_to_texture` call panics.
            if let Some(window) = web_sys::window() {
                if let Some(document) = window.document() {
                    if let Ok(canvas_element) = document.create_element("canvas") {
                        let canvas: HtmlCanvasElement = canvas_element.unchecked_into();
                        canvas.set_width(width);
                        canvas.set_height(height);
                        
                        if let Ok(Some(context_obj)) = canvas.get_context("2d") {
                            let context: CanvasRenderingContext2d = context_obj.unchecked_into();
                            if context.draw_image_with_image_bitmap(&bitmap, 0.0, 0.0).is_ok() {
                                if let Ok(image_data) = context.get_image_data(0.0, 0.0, width as f64, height as f64) {
                                    let data = image_data.data();
                                    queue.write_texture(
                                        wgpu::ImageCopyTexture {
                                            texture: &texture,
                                            mip_level: 0,
                                            origin: wgpu::Origin3d::ZERO,
                                            aspect: wgpu::TextureAspect::All,
                                        },
                                        &data,
                                        wgpu::ImageDataLayout {
                                            offset: 0,
                                            bytes_per_row: Some(4 * width),
                                            rows_per_image: Some(height),
                                        },
                                        size,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Generate mipmaps if pipeline is provided
        if let (Some(pipeline), Some(layout)) = (mipmap_pipeline, mipmap_bind_group_layout) {
            if mip_level_count > 1 {
                State::generate_mipmaps(
                    device,
                    queue,
                    &texture,
                    format,
                    width,
                    height,
                    mip_level_count,
                    pipeline,
                    layout,
                );
            }
        }

        Self { view: texture.create_view(&wgpu::TextureViewDescriptor::default()) }
    }

    pub fn single_pixel(device: &wgpu::Device, queue: &wgpu::Queue, color: [u8; 4], is_srgb: bool) -> Self {
        let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
        let format = if is_srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pixel Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &color,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            size,
        );

        Self { view: texture.create_view(&wgpu::TextureViewDescriptor::default()) }
    }
}

// --- Mesh & Model ---

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    material_bind_group: wgpu::BindGroup,
    // Cache indices to update bind group later
    diffuse_index: Option<usize>,
    normal_index: Option<usize>,
    mr_index: Option<usize>,
    // Current views
    diffuse_view: Rc<wgpu::TextureView>,
    normal_view: Rc<wgpu::TextureView>,
    mr_view: Rc<wgpu::TextureView>,
    sampler: Rc<wgpu::Sampler>,
    center: Vec3, // For sorting transparent meshes
    aabb_min: Vec3, // Axis-aligned bounding box minimum
    aabb_max: Vec3, // Axis-aligned bounding box maximum
}

struct Model {
    opaque_meshes: Vec<Mesh>,
    transparent_meshes: Vec<Mesh>,
}

// Optimized: Generates at origin (0,0,0). Position applied via Uniform.
fn create_sphere_mesh(device: &wgpu::Device, queue: &wgpu::Queue, radius: f32, segments: u32, material_layout: &wgpu::BindGroupLayout) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    
    for i in 0..=segments {
        let theta = (i as f32 / segments as f32) * std::f32::consts::PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        for j in 0..=segments {
            let phi = (j as f32 / segments as f32) * 2.0 * std::f32::consts::PI;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();
            let x = cos_phi * sin_theta;
            let y = cos_theta;
            let z = sin_phi * sin_theta;
            positions.push([x * radius, y * radius, z * radius]); // Origin
            normals.push([x, y, z]);
            uvs.push([j as f32 / segments as f32, i as f32 / segments as f32]);
        }
    }
    
    for i in 0..segments {
        for j in 0..segments {
            let first = (i * (segments + 1) + j) as u32;
            let second = first + segments + 1;
            indices.push(first); indices.push(second); indices.push(first + 1);
            indices.push(second); indices.push(second + 1); indices.push(first + 1);
        }
    }
    
    let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
    let mut vertices = Vec::new();
    for i in 0..positions.len() {
        vertices.push(ModelVertex {
            position: positions[i],
            normal: normals[i],
            tex_coord: uvs[i],
            tangent: tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
        });
    }
    
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Blob Vertex Buffer"), contents: bytemuck::cast_slice(&vertices), usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Blob Index Buffer"), contents: bytemuck::cast_slice(&indices), usage: wgpu::BufferUsages::INDEX,
    });
    
    let emissive_texture = Texture::single_pixel(device, queue, [255, 255, 200, 255], true);
    let smooth_texture = Texture::single_pixel(device, queue, [0, 0, 0, 255], false);
    let white_normal = Texture::single_pixel(device, queue, [128, 128, 255, 255], false);
    
    let sampler = Rc::new(device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    }));
    
    let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: material_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&emissive_texture.view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&white_normal.view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&smooth_texture.view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
        label: None,
    });
    
    Mesh {
        vertex_buffer, index_buffer, num_indices: indices.len() as u32, material_bind_group,
        diffuse_index: None, normal_index: None, mr_index: None,
        diffuse_view: Rc::new(emissive_texture.view), normal_view: Rc::new(white_normal.view), mr_view: Rc::new(smooth_texture.view),
        sampler, center: Vec3::ZERO, aabb_min: Vec3::splat(-radius), aabb_max: Vec3::splat(radius),
    }
}

fn compute_tangents(positions: &[[f32; 3]], normals: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]) -> Vec<[f32; 4]> {
    let mut tan1 = vec![Vec3::ZERO; positions.len()];
    let mut tan2 = vec![Vec3::ZERO; positions.len()];

    for i in (0..indices.len()).step_by(3) {
        let i1 = indices[i] as usize;
        let i2 = indices[i+1] as usize;
        let i3 = indices[i+2] as usize;

        let v1 = Vec3::from(positions[i1]);
        let v2 = Vec3::from(positions[i2]);
        let v3 = Vec3::from(positions[i3]);

        let w1 = vec2(uvs[i1][0], uvs[i1][1]);
        let w2 = vec2(uvs[i2][0], uvs[i2][1]);
        let w3 = vec2(uvs[i3][0], uvs[i3][1]);

        let x1 = v2.x - v1.x; let x2 = v3.x - v1.x;
        let y1 = v2.y - v1.y; let y2 = v3.y - v1.y;
        let z1 = v2.z - v1.z; let z2 = v3.z - v1.z;

        let s1 = w2.x - w1.x; let s2 = w3.x - w1.x;
        let t1 = w2.y - w1.y; let t2 = w3.y - w1.y;

        let r_denom = s1 * t2 - s2 * t1;
        let r = if r_denom.abs() < 1e-6 { 0.0 } else { 1.0 / r_denom };

        let sdir = vec3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
        let tdir = vec3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

        tan1[i1] += sdir; tan1[i2] += sdir; tan1[i3] += sdir;
        tan2[i1] += tdir; tan2[i2] += tdir; tan2[i3] += tdir;
    }

    let mut tangents = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        let n = Vec3::from(normals[i]);
        let t = tan1[i];
        
        // Gram-Schmidt orthogonalize
        let xyz = (t - n * n.dot(t)).normalize_or_zero();
        
        // Calculate handedness
        let w = if n.cross(t).dot(tan2[i]) < 0.0 { -1.0 } else { 1.0 };
        tangents.push([xyz.x, xyz.y, xyz.z, w]);
    }
    tangents
}

// --- State ---

#[wasm_bindgen]
pub struct State {
    surface: wgpu::Surface<'static>,
    device: Rc<wgpu::Device>,
    queue: Rc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    // MSAA resolve target (only used when SAMPLE_COUNT > 1)
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    sample_count: u32,
    opaque_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    blob_pipeline: wgpu::RenderPipeline, // NEW PIPELINE for blob
    mipmap_pipeline_linear: Rc<wgpu::RenderPipeline>,
    mipmap_pipeline_srgb: Rc<wgpu::RenderPipeline>,
    mipmap_bind_group_layout: Rc<wgpu::BindGroupLayout>,
    
    // Resources
    scene_uniform: SceneUniform,
    scene_buffer: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    
    // Camera State (Polar coordinates)
    camera_target: Vec3,
    camera_radius: f32,
    camera_azimuth: f32,
    camera_polar: f32,
    
    model_center: Vec3, // Center of the model
    model_extent: f32, // Maximum extent of the model (for orbital path sizing)
    
    light_pos_3d: Vec3, cursor_light_pos_3d: Vec3, cursor_light_active: bool, blob_light_pos_3d: Vec3,
    is_dark_theme: bool,
    
    // Base colors for lighting (modified by theme or manual override)
    base_light_color: [f32; 4],
    base_sky_color: [f32; 4],
    
    blob_exists: bool,
    blob_position: Vec3,
    blob_target_position: Vec3,
    blob_light_enabled: bool,
    blob_mesh: Mesh, // OPTIMIZATION: Stored by value, created once
    blob_dragging: bool,
    blob_drag_depth: f32,
    
    // Model System
    model: Option<Model>,
    material_layout: wgpu::BindGroupLayout,
    texture_cache: HashMap<usize, Rc<wgpu::TextureView>>,
    tx: Sender<AssetMessage>,
    rx: Receiver<AssetMessage>,
}

impl State {
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
            sample_count, // Match MSAA count
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

    // Generate mipmaps for a texture using a render pass
    // This is compatible with both WebGPU and WebGL2
    // Static method (doesn't need State instance)
    fn generate_mipmaps(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        mip_level_count: u32,
        mipmap_pipeline: &wgpu::RenderPipeline,
        mipmap_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        if mip_level_count <= 1 {
            return;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mipmap Generation Encoder"),
        });

        // Generate each mip level from the previous one
        for mip_level in 1..mip_level_count {
            let _mip_width = (width >> mip_level).max(1);
            let _mip_height = (height >> mip_level).max(1);

            // Create view for source (previous mip level)
            let source_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Mipmap Source {}", mip_level - 1)),
                format: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip_level - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            });

            // Create view for destination (current mip level)
            // CRITICAL: The view format must match the pipeline target format exactly
            // For sRGB textures, we need to explicitly set the format to match the pipeline
            let dest_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Mipmap Dest {}", mip_level)),
                format: Some(format), // Explicitly set format to match pipeline
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            });

            // Create sampler for source texture
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Mipmap Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            // Create bind group for mipmap generation
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: mipmap_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some(&format!("Mipmap Bind Group {}", mip_level)),
            });

            // Render to mip level
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("Mipmap Generation Pass {}", mip_level)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &dest_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                render_pass.set_pipeline(mipmap_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    fn update_scene_uniforms(&mut self) {
        self.queue.write_buffer(&self.scene_buffer, 0, bytemuck::cast_slice(&[self.scene_uniform]));
    }

    fn update_camera_uniforms(&mut self) {
        let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
        let y = self.camera_radius * self.camera_polar.cos();
        let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
        let pos = vec3(x, y, z) + self.camera_target;
        
        let view = Mat4::look_at_rh(pos, self.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), self.config.width as f32 / self.config.height as f32, 0.1, 100.0);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        self.scene_uniform.camera = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, 1.0],
        };
    }

    // Frustum culling: Check if AABB is visible in camera frustum
    fn is_aabb_in_frustum(&self, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        // Get camera matrices
        let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
        let y = self.camera_radius * self.camera_polar.cos();
        let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
        let cam_pos = vec3(x, y, z) + self.camera_target;
        
        let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), self.config.width as f32 / self.config.height as f32, 0.1, 100.0);
        let view_proj = proj * view;

        // Extract frustum planes from view-projection matrix
        // For column-major matrices, we extract planes from rows (transposed access)
        // Frustum planes: left, right, bottom, top, near, far
        let m = view_proj.to_cols_array_2d();
        let planes = [
            // Left plane (row 3 + row 0)
            vec4(m[3][0] + m[0][0], m[3][1] + m[0][1], m[3][2] + m[0][2], m[3][3] + m[0][3]),
            // Right plane (row 3 - row 0)
            vec4(m[3][0] - m[0][0], m[3][1] - m[0][1], m[3][2] - m[0][2], m[3][3] - m[0][3]),
            // Bottom plane (row 3 + row 1)
            vec4(m[3][0] + m[1][0], m[3][1] + m[1][1], m[3][2] + m[1][2], m[3][3] + m[1][3]),
            // Top plane (row 3 - row 1)
            vec4(m[3][0] - m[1][0], m[3][1] - m[1][1], m[3][2] - m[1][2], m[3][3] - m[1][3]),
            // Near plane (row 3 + row 2)
            vec4(m[3][0] + m[2][0], m[3][1] + m[2][1], m[3][2] + m[2][2], m[3][3] + m[2][3]),
            // Far plane (row 3 - row 2)
            vec4(m[3][0] - m[2][0], m[3][1] - m[2][1], m[3][2] - m[2][2], m[3][3] - m[2][3]),
        ];

        // Normalize planes
        let planes: Vec<Vec4> = planes.iter().map(|p| {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if len > 0.0001 {
                *p / len
            } else {
                *p
            }
        }).collect();

        // Test AABB against each frustum plane
        for plane in planes {
            // Find the positive vertex (P-vertex) of the AABB for this plane
            let p_vertex = vec3(
                if plane.x >= 0.0 { aabb_max.x } else { aabb_min.x },
                if plane.y >= 0.0 { aabb_max.y } else { aabb_min.y },
                if plane.z >= 0.0 { aabb_max.z } else { aabb_min.z },
            );

            // If P-vertex is outside the plane, the AABB is outside the frustum
            let distance = plane.x * p_vertex.x + plane.y * p_vertex.y + plane.z * p_vertex.z + plane.w;
            if distance < 0.0 {
                return false;
            }
        }

        true
    }
}

#[wasm_bindgen]
impl State {
    #[wasm_bindgen(js_name = "setTheme")]
    pub fn set_theme(&mut self, is_dark: bool) {
        self.is_dark_theme = is_dark;
        
        if is_dark {
            // Dark theme defaults - Deep Space / Cyber
            self.base_light_color = [0.96, 0.98, 1.0, 1.0]; // Crisp cool white
            self.base_sky_color = [0.08, 0.1, 0.15, 1.0]; // Deep midnight blue
        } else {
            // Light theme defaults - Dreamy / Ethereal
            self.base_light_color = [1.0, 0.96, 0.92, 1.0]; // Soft warm sun
            self.base_sky_color = [0.92, 0.94, 0.98, 1.0]; // Pale blue sky
        }

        web_sys::console::log_1(&format!("Theme updated to: {}", if is_dark { "Dark" } else { "Light" }).into());
        // Update lighting immediately based on theme
        self.update_theme_lighting();
    }
    
    #[wasm_bindgen(js_name = "setEnvironmentLight")]
    pub fn set_environment_light(&mut self, sky_r: f32, sky_g: f32, sky_b: f32, light_r: f32, light_g: f32, light_b: f32) {
        self.base_sky_color = [sky_r, sky_g, sky_b, 1.0];
        self.base_light_color = [light_r, light_g, light_b, 1.0];
        self.update_theme_lighting();
    }

    #[wasm_bindgen(js_name = "setCursorLightActive")]
    pub fn set_cursor_light_active(&mut self, active: bool) {
        self.cursor_light_active = active;
        
        // Immediately recalculate the final light position based on current state.
        // By not resetting cursor_light_pos_3d here, it will retain its last known
        // position, preventing a visual jump when the light is toggled on.
        let blob_active = self.blob_exists && self.blob_light_enabled;
        let cursor_active = self.cursor_light_active;
        
        if blob_active && cursor_active {
            // Both active - blend their positions
            self.light_pos_3d = self.blob_light_pos_3d * 0.6 + self.cursor_light_pos_3d * 0.4;
        } else if blob_active {
            // Only blob light active - use blob position only
            self.light_pos_3d = self.blob_light_pos_3d;
        } else if cursor_active {
            // Only cursor light active
            self.light_pos_3d = self.cursor_light_pos_3d;
        } else {
            // Neither active - use default position (center, above model)
            self.light_pos_3d = vec3(0.0, 5.0, 0.0);
        }
        // Update lighting immediately and write to buffer
        self.update_theme_lighting();
    }
    
    // Update lighting colors based on current state (base colors + intensity boosts)
    // Determines which light to use: blob light (if exists and enabled) OR cursor light (if dynamic light active)
    fn update_theme_lighting(&mut self) {
        let light_pos = self.light_pos_3d;
        
        let blob_active = self.blob_exists && self.blob_light_enabled;
        let cursor_active = self.cursor_light_active;
        
        // If NO light source is active, zero out the intensity completely
        if !blob_active && !cursor_active {
            self.scene_uniform.light = LightUniform {
                position: [light_pos.x, light_pos.y, light_pos.z, 1.0],
                color: [0.0, 0.0, 0.0, 0.0],
                sky_color: [0.0, 0.0, 0.0, 1.0], // Also kill ambient to be sure
            };
            return;
        }

        // Increase intensity for both blob and cursor lights
        // Base boost: 2.0x for all lights
        // Additional boost for blob: 0.5x
        // Additional boost for cursor light: 1.1x (makes dynamic light more intense)
        let base_intensity_boost = 2.0; // Base boost for all lights
        let blob_additional_boost = if blob_active { 0.5 } else { 0.0 }; // Extra boost for blob
        let cursor_additional_boost = if cursor_active && !blob_active { 1.1 } else { 0.0 }; // Extra boost for cursor light when it's the only active light
        let intensity_boost = base_intensity_boost + blob_additional_boost + cursor_additional_boost;
        
        // Use stored base colors (set by theme or JS override)
        let base_color = self.base_light_color;
        
        // Apply intensity boost to the main light color
        let max_intensity = if self.is_dark_theme { 3.0 } else { 2.5 };
        
        // Lights Out Logic: If Sun/Moon (Blob) is gone, kill ambient light
        // This makes the dynamic light the ONLY source of visibility
        let ambient_mult = if blob_active { 1.0 } else { 0.05 }; // 5% ambient if sun is gone (moonlight remnant)

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
    

    #[wasm_bindgen(js_name = "loadModelFromBytes")]
    pub fn load_model_from_bytes(&mut self, bytes: &[u8]) {
        // HACK: Patch "extensionsRequired" to "extensionsOptional" to bypass gltf validation
        // for unsupported extensions like KHR_texture_transform.
        let mut modified_bytes = std::borrow::Cow::Borrowed(bytes);
        if bytes.len() >= 20 && &bytes[0..4] == b"glTF" {
            let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap_or([0,0,0,0])) as usize;
            let chunk_type = &bytes[16..20];
            if chunk_type == b"JSON" && bytes.len() >= 20 + json_len {
                let json_slice = &bytes[20..20+json_len];
                let key = b"\"extensionsRequired\"";
                if let Some(pos) = json_slice.windows(key.len()).position(|w| w == key) {
                    let mut owned = bytes.to_vec();
                    let replacement = b"\"extensionsOptional\"";
                    let abs_pos = 20 + pos;
                    owned[abs_pos..abs_pos + key.len()].copy_from_slice(replacement);
                    modified_bytes = std::borrow::Cow::Owned(owned);
                    web_sys::console::warn_1(&"Patched GLB: Bypassed extensionsRequired validation (KHR_texture_transform likely)".into());
                }
            }
        }

        let result = gltf::Gltf::from_slice(&modified_bytes);
        match result {
            Ok(gltf) => {
                let document = gltf.document;
                let blob = gltf.blob.unwrap_or_default(); // Binary chunk
                
                let mut opaque_meshes = Vec::new();
                let mut transparent_meshes = Vec::new();
                let mut requested_textures = HashSet::new();

                // Default Textures
                let white_tex_srgb = Texture::single_pixel(&self.device, &self.queue, [255, 255, 255, 255], true);
                let white_tex_linear = Texture::single_pixel(&self.device, &self.queue, [255, 255, 255, 255], false);
                // Clay PBR Default: Occlusion=1.0 (R), Roughness=1.0 (G), Metallic=0.0 (B)
                let clay_mr_tex = Texture::single_pixel(&self.device, &self.queue, [255, 255, 0, 255], false);
                let normal_tex = Texture::single_pixel(&self.device, &self.queue, [128, 128, 255, 255], false);
                
                let white_view_srgb = Rc::new(white_tex_srgb.view);
                let white_view_linear = Rc::new(white_tex_linear.view);
                let clay_mr_view = Rc::new(clay_mr_tex.view);
                let normal_view = Rc::new(normal_tex.view);
                
                // High-quality sampler with anisotropic filtering
                let sampler = Rc::new(self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("Texture Sampler"),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    lod_min_clamp: 0.0,
                    lod_max_clamp: 100.0,
                    // Anisotropic filtering (16x is max, but 8x is usually sufficient)
                    // Note: WebGL2 may not support this, so it will fall back gracefully
                    anisotropy_clamp: 8,
                    compare: None,
                    border_color: None,
                }));

                for scene in document.scenes() {
                    for node in scene.nodes() {
                        let mut stack = vec![(node, Mat4::IDENTITY)];
                        while let Some((node, parent_transform)) = stack.pop() {
                            let (t, r, s) = node.transform().decomposed();
                            let local = Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));
                            let world = parent_transform * local;

                            if let Some(mesh) = node.mesh() {
                                for primitive in mesh.primitives() {
                                    let reader = primitive.reader(|buffer| {
                                        match buffer.source() {
                                            gltf::buffer::Source::Bin => Some(blob.as_slice()),
                                            _ => None
                                        }
                                    });

                                    let positions: Vec<[f32; 3]> = reader.read_positions().map(|i| i.collect()).unwrap_or_default();
                                    let normals: Vec<[f32; 3]> = reader.read_normals().map(|i| i.collect()).unwrap_or_else(|| vec![[0.0; 3]; positions.len()]);
                                    let tex_coords: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|v| v.into_f32().collect()).unwrap_or_else(|| vec![[0.0; 2]; positions.len()]);
                                    let indices: Vec<u32> = reader.read_indices().map(|i| i.into_u32().collect()).unwrap_or_else(|| (0..positions.len() as u32).collect());
                                    
                                    if positions.is_empty() { continue; }

                                    // Tangents
                                    let mut tangents: Vec<[f32; 4]> = reader.read_tangents().map(|i| i.collect()).unwrap_or_default();
                                    if tangents.is_empty() {
                                        tangents = compute_tangents(&positions, &normals, &tex_coords, &indices);
                                    }

                                    let mut min = vec3(f32::MAX, f32::MAX, f32::MAX);
                                    let mut max = vec3(f32::MIN, f32::MIN, f32::MIN);

                                    let vertices: Vec<ModelVertex> = positions.iter()
                                        .zip(normals.iter())
                                        .zip(tex_coords.iter())
                                        .zip(tangents.iter())
                                        .map(|(((p, n), uv), t)| {
                                            let pw = world * vec4(p[0], p[1], p[2], 1.0);
                                            min = min.min(pw.truncate());
                                            max = max.max(pw.truncate());
                                            let nw = world.transform_vector3(Vec3::from(*n)).normalize();
                                            let tw = world.transform_vector3(Vec3::new(t[0], t[1], t[2])).normalize();
                                            ModelVertex {
                                                position: [pw.x, pw.y, pw.z],
                                                normal: [nw.x, nw.y, nw.z],
                                                tex_coord: *uv,
                                                tangent: [tw.x, tw.y, tw.z, t[3]],
                                            }
                                        })
                                        .collect();

                                    let center = (min + max) * 0.5;

                                    let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                        label: Some("Mesh Vertex Buffer"),
                                        contents: bytemuck::cast_slice(&vertices),
                                        usage: wgpu::BufferUsages::VERTEX,
                                    });
                                    
                                    let i_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                        label: Some("Mesh Index Buffer"),
                                        contents: bytemuck::cast_slice(&indices),
                                        usage: wgpu::BufferUsages::INDEX,
                                    });

                                    // Material Handling
                                    let mat = primitive.material();
                                    let pbr = mat.pbr_metallic_roughness();
                                    
                                    // Texture Loading Helper
                                    let mut process_tex = |tex: Option<gltf::Texture>, type_id: u32, is_srgb: bool| -> Option<usize> {
                                        if let Some(texture) = tex {
                                            if let gltf::image::Source::View { view, mime_type } = texture.source().source() {
                                                let key = view.offset();
                                                if !requested_textures.contains(&key) {
                                                    requested_textures.insert(key);
                                                    let start = view.offset();
                                                    let end = start + view.length();
                                                    if end <= blob.len() {
                                                        let img_data = blob[start..end].to_vec();
                                                        let mime = mime_type.to_string();
                                                        let tx = self.tx.clone();
                                                        let dev = self.device.clone();
                                                        let q = self.queue.clone();
                                                        let mipmap_pipeline_linear = self.mipmap_pipeline_linear.clone();
                                                        let mipmap_pipeline_srgb = self.mipmap_pipeline_srgb.clone();
                                                        let mipmap_layout = self.mipmap_bind_group_layout.clone();
                                                        
                                                        wasm_bindgen_futures::spawn_local(async move {
                                                            let a = Array::new();
                                                            let u = unsafe { Uint8Array::view(&img_data) };
                                                            a.push(&u);
                                                            let props = BlobPropertyBag::new();
                                                            props.set_type(&mime);
                                                            let b = Blob::new_with_u8_array_sequence_and_options(&a, &props).unwrap();
                                                            let w = web_sys::window().unwrap();
                                                            if let Ok(bmp_val) = JsFuture::from(w.create_image_bitmap_with_blob(&b).unwrap()).await {
                                                                let bmp: ImageBitmap = bmp_val.unchecked_into();
                                                                let pipeline = if is_srgb { &mipmap_pipeline_srgb } else { &mipmap_pipeline_linear };
                                                                let t = Texture::from_bitmap(&dev, &q, bmp, is_srgb, Some(pipeline), Some(&mipmap_layout));
                                                                let _ = tx.send(AssetMessage::TextureLoaded { image_index: key, texture_type: type_id, texture: t });
                                                            }
                                                        });
                                                    }
                                                }
                                                return Some(key);
                                            }
                                        }
                                        None
                                    };

                                    let diff_idx = process_tex(pbr.base_color_texture().map(|t| t.texture()), 0, true);
                                    let norm_idx = process_tex(mat.normal_texture().map(|t| t.texture()), 1, false);
                                    let mr_idx = process_tex(pbr.metallic_roughness_texture().map(|t| t.texture()), 2, false);

                                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        layout: &self.material_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&white_view_srgb) },
                                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_view) },
                                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
                                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&clay_mr_view) }, // Use Clay PBR default
                                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&sampler) },
                                        ],
                                        label: Some("Material Bind Group"),
                                    });

                                    let mesh = Mesh {
                                        vertex_buffer: v_buf,
                                        index_buffer: i_buf,
                                        num_indices: indices.len() as u32,
                                        material_bind_group: bind_group,
                                        diffuse_index: diff_idx,
                                        normal_index: norm_idx,
                                        mr_index: mr_idx,
                                        diffuse_view: white_view_srgb.clone(),
                                        normal_view: normal_view.clone(),
                                        mr_view: clay_mr_view.clone(),
                                        sampler: sampler.clone(),
                                        center,
                                        aabb_min: min,
                                        aabb_max: max,
                                    };

                                    match mat.alpha_mode() {
                                        gltf::material::AlphaMode::Blend => transparent_meshes.push(mesh),
                                        _ => opaque_meshes.push(mesh),
                                    }
                                }
                            }
                            for child in node.children() { stack.push((child, world)); }
                        }
                    }
                }
                
                // Calculate overall model center and extent from all meshes
                let mut model_min = vec3(f32::MAX, f32::MAX, f32::MAX);
                let mut model_max = vec3(f32::MIN, f32::MIN, f32::MIN);
                
                // Use AABB from meshes for accurate extent calculation
                for mesh in opaque_meshes.iter().chain(transparent_meshes.iter()) {
                    model_min = model_min.min(mesh.aabb_min);
                    model_max = model_max.max(mesh.aabb_max);
                }
                
                // If we have meshes, calculate center and extent; otherwise use defaults
                if !opaque_meshes.is_empty() || !transparent_meshes.is_empty() {
                    self.model_center = (model_min + model_max) * 0.5;
                    // Calculate maximum extent (largest dimension)
                    let extent = model_max - model_min;
                    self.model_extent = extent.x.max(extent.y.max(extent.z));
                } else {
                    self.model_center = Vec3::ZERO;
                    self.model_extent = 2.0; // Default extent
                }
                
                self.model = Some(Model { opaque_meshes, transparent_meshes });
                web_sys::console::log_1(&format!("Model loaded with textures! Center: {:?}", self.model_center).into());
            },
            Err(e) => {
                web_sys::console::error_1(&format!("Failed to parse GLB: {:?}", e).into());
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            
            let (depth_texture, depth_view) = State::create_depth_texture(&self.device, &self.config, self.sample_count);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
            
            // Recreate MSAA texture if needed
            if let Some((msaa_texture, msaa_view)) = State::create_msaa_texture(&self.device, &self.config, self.sample_count) {
                self.msaa_texture = Some(msaa_texture);
                self.msaa_view = Some(msaa_view);
            } else {
                self.msaa_texture = None;
                self.msaa_view = None;
            }
            
            self.update_camera_uniforms();
        }
    }

    pub fn update_camera(&mut self, dx: f32, dy: f32, zoom: f32) {
        let sensitivity = 0.005;
        self.camera_azimuth -= dx * sensitivity;
        self.camera_polar = (self.camera_polar - dy * sensitivity).clamp(0.01, std::f32::consts::PI - 0.01);
        self.camera_radius = (self.camera_radius + zoom * 0.002 * self.camera_radius).clamp(0.5, 50.0);
        self.update_camera_uniforms();
    }

    pub fn update(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) {
        // Update cursor light position for 3D scene (only if cursor light is active)
        // Unproject mouse position to 3D space to place light at exact mouse position
        if self.cursor_light_active {
            // Convert mouse to normalized device coordinates (-1 to 1)
            let mouse_norm_x = (mouse_x / screen_width) * 2.0 - 1.0;
            let mouse_norm_y = 1.0 - (mouse_y / screen_height) * 2.0; // Flip Y
            
            // Get camera matrices
            let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
            let y = self.camera_radius * self.camera_polar.cos();
            let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
            let cam_pos = vec3(x, y, z) + self.camera_target;
            
            let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
            let proj = Mat4::perspective_rh(45.0_f32.to_radians(), screen_width / screen_height, 0.1, 100.0);
            let inv_view_proj = (proj * view).inverse();
            
            // Create ray from camera through mouse position
            // Unproject near and far points to create a ray
            let near_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 0.0, 1.0);
            let far_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 1.0, 1.0);
            
            let near_world = vec3(near_point.x, near_point.y, near_point.z) / near_point.w;
            let far_world = vec3(far_point.x, far_point.y, far_point.z) / far_point.w;
            let ray_dir = (far_world - near_world).normalize();
            
            // Place light at a fixed distance along the ray from the camera
            // Closer distance creates a wider light cone effect (light spreads more)
            // This makes the dynamic light illuminate a wider area
            let light_distance = 3.0; // Reduced from 5.0 - closer light = wider cone effect
            let light_pos = near_world + ray_dir * light_distance;
            
            // Place light at this 3D position
            self.cursor_light_pos_3d = light_pos;
        }
        
        // If dragging blob, use screen-space dragging that follows mouse naturally
        // This provides intuitive dragging that works on both mobile and desktop
        if self.blob_dragging && self.blob_exists {
            // Unproject mouse position to 3D world space
            let mouse_norm_x = (mouse_x / screen_width) * 2.0 - 1.0;
            let mouse_norm_y = 1.0 - (mouse_y / screen_height) * 2.0; // Flip Y
            
            // Get camera position and matrices
            let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
            let y = self.camera_radius * self.camera_polar.cos();
            let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
            let cam_pos = vec3(x, y, z) + self.camera_target;
            
            let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
            let proj = Mat4::perspective_rh(45.0_f32.to_radians(), screen_width / screen_height, 0.1, 100.0);
            let inv_view_proj = (proj * view).inverse();
            
            // Use stable depth from when dragging started to prevent jumping
            // If depth is 0.0 (not initialized), calculate it from current blob position
            let depth = if self.blob_drag_depth > 0.0 {
                self.blob_drag_depth
                } else {
                // Calculate depth from current blob position (project to NDC)
                let blob_clip = proj * view * vec4(self.blob_position.x, self.blob_position.y, self.blob_position.z, 1.0);
                let blob_ndc_z = blob_clip.z / blob_clip.w;
                blob_ndc_z.clamp(0.0, 1.0)
            };
            
            // Unproject mouse position to 3D world space at the stable depth
            // This creates a plane at the blob's depth that the mouse moves on
            let mouse_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, depth, 1.0);
            let mouse_world = vec3(mouse_point.x, mouse_point.y, mouse_point.z) / mouse_point.w;
            
            // Only apply soft constraints to prevent going through the model
            // Don't force it onto a circular path - allow straight-line movement
            let to_model = mouse_world - self.model_center;
            let base_radius = (self.model_extent * 0.5).max(3.0);
            let min_radius = base_radius * 0.5; // Allow closer for more freedom
            let distance = to_model.length();
            
            // Only constrain if too close to model center (prevent going through model)
            // Don't constrain maximum distance - allow free movement
            let final_position = if distance < min_radius {
                // If too close, push it out to minimum radius
                if distance > 0.001 {
                    let direction = to_model / distance;
                    self.model_center + direction * min_radius
                } else {
                    // If at center, place at default position
                    self.model_center + vec3(min_radius, 0.0, 0.0)
                }
            } else {
                // Free movement - follow mouse directly
                mouse_world
            };
            
            // Update target position directly - will be applied with minimal smoothing during drag
            self.blob_target_position = final_position;
        }
        
        // Smooth interpolation towards target position (every frame, even when not dragging)
        if self.blob_exists {
            
            // Use adaptive smoothing: much faster when dragging for immediate response, slower when not dragging
            let smoothing_factor = if self.blob_dragging {
                0.8  // Much faster when dragging - almost direct following for responsiveness
            } else {
                0.15 // Slower when not dragging - smooth, natural movement
            };
            let diff = self.blob_target_position - self.blob_position;
            self.blob_position = self.blob_position + diff * smoothing_factor;
            
            // PERFORMANCE FIX: Update Uniform Buffer instead of recreating Mesh!
            // Use base_light_color so the blob looks like the light it emits (Sun/Moon color)
            self.scene_uniform.blob = BlobUniform {
                position: [self.blob_position.x, self.blob_position.y, self.blob_position.z, 0.0],
                color: self.base_light_color,
            };
            
            // Update blob light position (always track blob position)
            self.blob_light_pos_3d = self.blob_position;
        }
        // Light position blending is handled in render() function
        // This ensures both blob and cursor lights can contribute simultaneously
    }
    
    // Update blob Y position (for scroll during drag)
    // This adjusts the distance from model by changing Y, which affects the horizontal radius
    #[wasm_bindgen(js_name = "updateBlobY")]
    pub fn update_blob_y(&mut self, delta_y: f32) {
        if self.blob_exists && self.blob_dragging {
            // Adjust Y position (closer/further from model)
            // Positive delta = move up (further), negative = move down (closer)
            let y_change = -delta_y * 0.005; // Scale scroll sensitivity (negative for natural scroll direction)
            let new_y = (self.blob_target_position.y + y_change).clamp(
                self.model_center.y - 2.0, // Can go below model
                self.model_center.y + 10.0  // Can go well above model
            );
            self.blob_target_position.y = new_y;
            
            // When Y changes, the plane locking in update() will adjust the horizontal position
            // to maintain the correct distance from model center
        }
    }
    
    // Spawn light blob at cursor position
    #[wasm_bindgen(js_name = "spawnBlob")]
    pub fn spawn_blob(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) {
        if !self.blob_exists {
            // Defer setting blob_exists to true until AFTER position is calculated
            // Immediately position blob at cursor
            let mouse_norm_x = (mouse_x / screen_width) * 2.0 - 1.0;
            let mouse_norm_y = 1.0 - (mouse_y / screen_height) * 2.0;

            let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
            let y = self.camera_radius * self.camera_polar.cos();
            let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
            let cam_pos = vec3(x, y, z) + self.camera_target;
            
            let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
            let proj = Mat4::perspective_rh(45.0_f32.to_radians(), screen_width / screen_height, 0.1, 100.0);
            let inv_view_proj = (proj * view).inverse();

            let near_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 0.0, 1.0);
            let far_point = inv_view_proj * vec4(mouse_norm_x, mouse_norm_y, 1.0, 1.0);
            let near_world = vec3(near_point.x, near_point.y, near_point.z) / near_point.w;
            let far_world = vec3(far_point.x, far_point.y, far_point.z) / far_point.w;
            let ray_dir = (far_world - near_world).normalize();
            
            // Place blob along ray, at a distance relative to model size
            let spawn_distance = self.model_extent * 1.5;
            let spawn_position = near_world + ray_dir * spawn_distance;

            self.blob_position = spawn_position;
            self.blob_target_position = spawn_position;
            self.blob_light_enabled = true;
            self.blob_dragging = false;
            self.blob_light_pos_3d = self.blob_position;

            // CRITICAL FIX: Update the scene uniform's blob position directly
            // before enabling rendering and writing to the buffer.
            self.scene_uniform.blob.position = [spawn_position.x, spawn_position.y, spawn_position.z, 0.0];
            
            // Now that position is set, enable rendering
            self.blob_exists = true;

            self.update_theme_lighting();
            self.update_scene_uniforms(); // This now writes the correct position
        }
    }
    
    // Despawn light blob
    #[wasm_bindgen(js_name = "despawnBlob")]
    pub fn despawn_blob(&mut self) {
        self.blob_exists = false;
        // self.blob_mesh = None; // OPTIMIZATION: Keep mesh
        self.blob_dragging = false;
        // Light position blending is handled in render() function
        // When blob is despawned, render() will automatically use cursor light
        self.update_theme_lighting();
    }
    
    // Toggle blob light on/off (independent of dynamic light)
    #[wasm_bindgen(js_name = "toggleBlobLight")]
    pub fn toggle_blob_light(&mut self) {
        if self.blob_exists {
            self.blob_light_enabled = !self.blob_light_enabled;
            // Update blob light position if enabled
            if self.blob_light_enabled {
                self.blob_light_pos_3d = self.blob_position;
            }
            // Light position blending is handled in render() function
            // This allows both blob and cursor lights to work independently
            self.update_theme_lighting();
        }
    }
    
    // Check if mouse is hovering over blob (for cursor feedback)
    #[wasm_bindgen(js_name = "isHoveringBlob")]
    pub fn is_hovering_blob(&self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.blob_exists {
            return false;
        }
        
        let blob_screen = self.project_3d_to_screen(self.blob_position);
        let mouse_norm_x = mouse_x / screen_width;
        let mouse_norm_y = mouse_y / screen_height;
        
        let dist = ((blob_screen[0] - mouse_norm_x).powi(2) + (blob_screen[1] - mouse_norm_y).powi(2)).sqrt();
        
        // Larger threshold (15% of screen) for easier interaction
        // Also check if blob is visible (not behind camera)
        dist < 0.15 && blob_screen[0] > 0.0 && blob_screen[0] < 1.0 && blob_screen[1] > 0.0 && blob_screen[1] < 1.0
    }
    
    // Start dragging blob
    #[wasm_bindgen(js_name = "startDragBlob")]
    pub fn start_drag_blob(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.blob_exists {
            return false;
        }
        
        // Inline hover detection to avoid borrowing conflicts
        let is_hovering = {
            let blob_screen = self.project_3d_to_screen(self.blob_position);
            let mouse_norm_x = mouse_x / screen_width;
            let mouse_norm_y = mouse_y / screen_height;
            
            let dist = ((blob_screen[0] - mouse_norm_x).powi(2) + (blob_screen[1] - mouse_norm_y).powi(2)).sqrt();
            
            // Larger threshold (15% of screen) for easier interaction
            // Also check if blob is visible (not behind camera)
            dist < 0.15 && blob_screen[0] > 0.0 && blob_screen[0] < 1.0 && blob_screen[1] > 0.0 && blob_screen[1] < 1.0
        };

        if is_hovering {
            self.blob_dragging = true;
            // Initialize target position to current position to prevent jumping
            self.blob_target_position = self.blob_position;
            
            // Calculate and store stable depth from current blob position
            // This prevents jumping when dragging starts
            let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
            let y = self.camera_radius * self.camera_polar.cos();
            let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
            let cam_pos = vec3(x, y, z) + self.camera_target;
            let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
            let proj = Mat4::perspective_rh(45.0_f32.to_radians(), screen_width / screen_height, 0.1, 100.0);
            let blob_clip = proj * view * vec4(self.blob_position.x, self.blob_position.y, self.blob_position.z, 1.0);
            let blob_ndc_z = blob_clip.z / blob_clip.w;
            self.blob_drag_depth = blob_ndc_z.clamp(0.0, 1.0);
            
            return true;
        }
        
        false
    }
    
    // Stop dragging blob
    #[wasm_bindgen(js_name = "stopDragBlob")]
    pub fn stop_drag_blob(&mut self) {
        self.blob_dragging = false;
        self.blob_drag_depth = 0.0; // Reset depth
    }
    
    // Check if click hits blob (for toggling light)
    #[wasm_bindgen(js_name = "checkBlobClick")]
    pub fn check_blob_click(&mut self, mouse_x: f32, mouse_y: f32, screen_width: f32, screen_height: f32) -> bool {
        if !self.blob_exists {
            return false;
        }
        
        // Inline hover detection to avoid borrowing conflicts
        let is_hovering = {
            let blob_screen = self.project_3d_to_screen(self.blob_position);
            let mouse_norm_x = mouse_x / screen_width;
            let mouse_norm_y = mouse_y / screen_height;
            
            let dist = ((blob_screen[0] - mouse_norm_x).powi(2) + (blob_screen[1] - mouse_norm_y).powi(2)).sqrt();
            
            // Larger threshold (15% of screen) for easier interaction
            // Also check if blob is visible (not behind camera)
            dist < 0.15 && blob_screen[0] > 0.0 && blob_screen[0] < 1.0 && blob_screen[1] > 0.0 && blob_screen[1] < 1.0
        };
        
        if is_hovering {
            // Only toggle if not already dragging (to avoid toggling on drag start)
            if !self.blob_dragging {
                self.toggle_blob_light();
            }
            return true;
        }
        
        false
    }
    
    // Helper: Project 3D position to screen coordinates (0-1 range)
    fn project_3d_to_screen(&self, world_pos: Vec3) -> [f32; 2] {
        let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
        let y = self.camera_radius * self.camera_polar.cos();
        let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
        let cam_pos = vec3(x, y, z) + self.camera_target;
        
        let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), self.config.width as f32 / self.config.height as f32, 0.1, 100.0);
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
    
    // Project 3D light position to 2D screen coordinates for CSS
    #[wasm_bindgen(js_name = "getLightScreenPos")]
    pub fn get_light_screen_pos(&self) -> Array {
        // Get current camera matrices
        let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
        let y = self.camera_radius * self.camera_polar.cos();
        let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
        let cam_pos = vec3(x, y, z) + self.camera_target;
        
        let view = Mat4::look_at_rh(cam_pos, self.camera_target, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), self.config.width as f32 / self.config.height as f32, 0.1, 100.0);
        let view_proj = proj * view;
        
        // Transform 3D light position to clip space
        let light_world = vec4(self.light_pos_3d.x, self.light_pos_3d.y, self.light_pos_3d.z, 1.0);
        let light_clip = view_proj * light_world;
        
        // Perspective divide to get NDC (Normalized Device Coordinates)
        let w = light_clip.w;
        if w.abs() < 0.0001 {
            // Behind camera or at infinity, return center
            let result = Array::new_with_length(2);
            result.set(0, JsValue::from_f64(0.5));
            result.set(1, JsValue::from_f64(0.5));
            return result;
        }
        
        let ndc_x = light_clip.x / w;
        let ndc_y = light_clip.y / w;
        
        // Convert NDC to screen coordinates (0-1 range)
        let screen_x = (ndc_x + 1.0) * 0.5;
        let screen_y = 1.0 - (ndc_y + 1.0) * 0.5; // Flip Y axis
        
        // Return as JS array
        let result = Array::new_with_length(2);
        result.set(0, JsValue::from_f64(screen_x as f64));
        result.set(1, JsValue::from_f64(screen_y as f64));
        result
    }

    // Project 3D blob position to 2D screen coordinates for CSS UI repulsion
    #[wasm_bindgen(js_name = "getBlobScreenPosition")]
    pub fn get_blob_screen_position(&self) -> Array {
        if !self.blob_exists {
            return Array::new(); // Return empty if no blob
        }
        
        let screen_pos = self.project_3d_to_screen(self.blob_position);
        
        // Return as JS array [x, y]
        let result = Array::new_with_length(2);
        result.set(0, JsValue::from_f64(screen_pos[0] as f64));
        result.set(1, JsValue::from_f64(screen_pos[1] as f64));
        result
    }

    pub fn render(&mut self, time: f32) {
        self.scene_uniform.time = time;

        // 0. Handle Async Texture Loads
        while let Ok(msg) = self.rx.try_recv() {
            if let Some(model) = &mut self.model {
                let AssetMessage::TextureLoaded { image_index, texture_type, texture } = msg;
                let view = Rc::new(texture.view);
                self.texture_cache.insert(image_index, view.clone());
                
                let update_mesh = |mesh: &mut Mesh| {
                    let mut update = false;
                    if texture_type == 0 && mesh.diffuse_index == Some(image_index) { mesh.diffuse_view = view.clone(); update = true; }
                    if texture_type == 1 && mesh.normal_index == Some(image_index) { mesh.normal_view = view.clone(); update = true; }
                    if texture_type == 2 && mesh.mr_index == Some(image_index) { mesh.mr_view = view.clone(); update = true; }
                    
                    if update {
                        mesh.material_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.material_layout,
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

                for mesh in &mut model.opaque_meshes { update_mesh(mesh); }
                for mesh in &mut model.transparent_meshes { update_mesh(mesh); }
            }
        }

        // 1. Update lighting (ensure it's updated every frame)
        // Both blob light and cursor light can be active simultaneously
        // When both are active, blend their positions to create combined lighting
        // Update blob light position if blob exists and is enabled
        if self.blob_exists && self.blob_light_enabled {
            self.blob_light_pos_3d = self.blob_position;
        }
        
        // Determine final light position based on which lights are active
        // Cursor light is only active when cursor_light_active is true (controlled by dynamic light toggle)
        // When blob is active, blend with cursor light ONLY if cursor is also active
        let blob_active = self.blob_exists && self.blob_light_enabled;
        let cursor_active = self.cursor_light_active;
        
        // Decouple blob light from cursor light. If blob is active, it's the ONLY light source.
        if blob_active {
            self.light_pos_3d = self.blob_light_pos_3d;
        } else if cursor_active {
            self.light_pos_3d = self.cursor_light_pos_3d;
        } else {
            // Fallback if no light is active
            self.light_pos_3d = vec3(0.0, 5.0, 0.0);
        }
        self.update_theme_lighting();

        // Update the consolidated scene uniform buffer before drawing
        self.update_scene_uniforms();

        // 3. Draw
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
            // Use MSAA texture if available, otherwise render directly to surface
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

            // Bind Global Scene Uniforms (Group 0)
            render_pass.set_bind_group(0, &self.scene_bind_group, &[]);

            // Draw Model Opaque
            // Draw Model Opaque
            if let Some(model) = &self.model {
                render_pass.set_pipeline(&self.opaque_pipeline);
                // Bindings 0, 1, 2 are already set above and valid for Opaque Pipeline too
                
                for mesh in &model.opaque_meshes {
                    // Frustum culling: skip meshes outside camera view
                    if !self.is_aabb_in_frustum(mesh.aabb_min, mesh.aabb_max) {
                        continue;
                    }
                    
                    render_pass.set_bind_group(1, &mesh.material_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }

                // Draw Model Transparent (Sorted by distance from camera)
                // Sort transparent meshes before rendering for correct alpha blending
                // Get camera position for sorting
                let x = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.sin();
                let y = self.camera_radius * self.camera_polar.cos();
                let z = self.camera_radius * self.camera_polar.sin() * self.camera_azimuth.cos();
                let cam_pos = vec3(x, y, z) + self.camera_target;
                
                // Create sorted list of visible transparent meshes with their distances
                let mut transparent_to_draw: Vec<(&Mesh, f32)> = model.transparent_meshes.iter()
                    .filter_map(|mesh| {
                        // Frustum culling: skip meshes outside camera view
                        if !self.is_aabb_in_frustum(mesh.aabb_min, mesh.aabb_max) {
                            return None;
                        }
                        // Calculate distance from camera to mesh center
                        let distance = (mesh.center - cam_pos).length_squared(); // Use squared distance for efficiency
                        Some((mesh, distance))
                    })
                    .collect();
                
                // Sort by distance (farthest first for correct back-to-front rendering)
                transparent_to_draw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                render_pass.set_pipeline(&self.transparent_pipeline);
                for (mesh, _) in transparent_to_draw {
                    render_pass.set_bind_group(1, &mesh.material_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }

            // Draw Blob if it exists
            if self.blob_exists {
                render_pass.set_pipeline(&self.blob_pipeline);
                // Bind scene uniforms at group 0 and material at group 1
                render_pass.set_bind_group(1, &self.blob_mesh.material_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.blob_mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.blob_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.blob_mesh.num_indices, 0, 0..1);
            }
            
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

#[wasm_bindgen(js_name = "startRenderer")]
#[allow(unreachable_code, unused_variables)]
pub async fn start_renderer(canvas: HtmlCanvasElement, is_mobile: bool) -> Result<State, JsValue> {
    #[cfg(feature = "console_error_panic_hook")]
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    web_sys::console::log_1(&format!("Initializing Advanced WGPU Renderer... Mobile: {}", is_mobile).into());

    let instance = wgpu::Instance::default();
    
    // For wgpu 0.20 on web, we must use SurfaceTarget::Canvas.
    // We use cfg gates to ensure this compiles on both WASM (where Canvas variant exists)
    // and Host (where it doesn't, preventing IDE errors).
    #[cfg(target_arch = "wasm32")]
    let target = wgpu::SurfaceTarget::Canvas(canvas.clone());
    
    #[cfg(not(target_arch = "wasm32"))]
    let target: wgpu::SurfaceTarget = unimplemented!("Not supported on non-WASM targets");

    let surface = instance.create_surface(target)
        .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {}", e)))?;
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false }).await.ok_or_else(|| JsValue::from_str("No adapter"))?;
    
    // Request device with more features if possible
    let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults();
    // Upgrade texture limits to device capabilities (crucial for Retina/4K)
    let adapter_limits = adapter.limits();
    required_limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
    
    // Explicitly zero out compute limits to avoid "value 65535 is better than allowed 0" error
    // caused by some browser/driver combinations reporting high limits but enforcing 0 for WebGL.
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
        .unwrap_or_else(|| surface_caps.formats.first().copied().unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb));

    let present_mode = surface_caps.present_modes.iter()
        .copied()
        .find(|&mode| mode == wgpu::PresentMode::Mailbox)
        .unwrap_or(wgpu::PresentMode::Fifo);

    // Use PreMultiplied alpha mode for transparency support
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
    
    // Detect backend and adjust MSAA sample count accordingly.
    // WebGL often doesn't support MSAA storage textures, causing a panic.
    // By checking the backend, we can disable MSAA for WebGL and keep it for WebGPU.
    let info = adapter.get_info();
    let actual_sample_count = if info.backend == wgpu::Backend::Gl {
        web_sys::console::warn_1(&"WebGL backend detected, disabling MSAA to prevent panic.".into());
        1
    } else if is_mobile {
        2 // Reduce MSAA for mobile
    } else {
        SAMPLE_COUNT
    };
    
    let (depth_texture, depth_view) = State::create_depth_texture(&device, &config, actual_sample_count);
    let (msaa_texture, msaa_view) = State::create_msaa_texture(&device, &config, actual_sample_count)
        .map(|(t, v)| (Some(t), Some(v)))
        .unwrap_or((None, None));

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    // --- Buffers & Layouts ---

    // 0. Scene Uniform (Consolidated)
    // Initial colors for Dark Theme (default)
    let base_light = [0.96, 0.98, 1.0, 1.0];
    let base_sky = [0.08, 0.1, 0.15, 1.0];

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

    // 1. Material
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

    // Pipeline Layout (Main)
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Main Pipeline Layout"),
        bind_group_layouts: &[&scene_layout, &material_layout],
        push_constant_ranges: &[],
    });

    // --- Pipelines ---

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

    // NEW: Blob Pipeline (Uses vs_blob entry point)
    let blob_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Blob Pipeline"), layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_blob", buffers: &[ModelVertex::desc()], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_model", targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
        primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
        multisample: wgpu::MultisampleState { count: actual_sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None
    });

    // Create static blob mesh (at 0,0,0)
    // SOTA Optimization: High-Poly Sphere (64 segments) for perfectly smooth silhouette
    let blob_mesh = create_sphere_mesh(&device, &queue, 0.5, 64, &material_layout);

    // Mipmap Generation Pipeline
    let mipmap_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    });

    let mipmap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mipmap Pipeline Layout"),
        bind_group_layouts: &[&mipmap_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Helper function to create mipmap pipelines for different formats
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

    let mipmap_pipeline_linear = create_mipmap_pipeline(wgpu::TextureFormat::Rgba8Unorm, "Mipmap Pipeline (Linear)");
    let mipmap_pipeline_srgb = create_mipmap_pipeline(wgpu::TextureFormat::Rgba8UnormSrgb, "Mipmap Pipeline (sRGB)");

    let (tx, rx) = flume::unbounded();

    Ok(State {
        surface, device: Rc::new(device), queue: Rc::new(queue), config, depth_texture, depth_view,
        msaa_texture, msaa_view, sample_count: actual_sample_count,
        opaque_pipeline, transparent_pipeline, blob_pipeline,
        mipmap_pipeline_linear: Rc::new(mipmap_pipeline_linear),
        mipmap_pipeline_srgb: Rc::new(mipmap_pipeline_srgb),
        mipmap_bind_group_layout: Rc::new(mipmap_bind_group_layout),
        scene_uniform, scene_buffer, scene_bind_group,
        camera_target: Vec3::ZERO, camera_radius: 10.0, camera_azimuth: 0.0, camera_polar: 1.57,
        light_pos_3d: vec3(2.0, 2.0, 2.0), cursor_light_pos_3d: vec3(2.0, 2.0, 2.0), cursor_light_active: false, blob_light_pos_3d: Vec3::ZERO,
        is_dark_theme: true, // Default to dark theme
        base_light_color: base_light,
        base_sky_color: base_sky,
        blob_exists: false, // Default hidden
        blob_position: vec3(0.0, 5.0, 0.0), // Start above center
        blob_target_position: vec3(0.0, 5.0, 0.0),
        blob_light_enabled: false,
        blob_mesh,
        blob_dragging: false,
        blob_drag_depth: 0.0,
        model: None, model_center: Vec3::ZERO, model_extent: 2.0, material_layout, texture_cache: HashMap::new(), tx, rx,
    })
}