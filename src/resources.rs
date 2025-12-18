use wgpu::util::DeviceExt;
use glam::{vec2, vec3, vec4, Mat4, Vec3, Vec4, Quat};
use std::rc::Rc;
use std::collections::HashSet;
use flume::Sender;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{ImageBitmap, Blob, BlobPropertyBag, HtmlCanvasElement, CanvasRenderingContext2d};
use js_sys::{Uint8Array, Array};
use std::panic;
use crate::uniforms::ModelVertex;

pub enum AssetMessage {
    TextureLoaded { 
        image_index: usize, 
        texture_type: u32, 
        texture: Texture 
    }
}

pub struct Texture {
    pub view: wgpu::TextureView,
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

        let mut uploaded = false;

        #[cfg(target_arch = "wasm32")]
        {
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

        if let (Some(pipeline), Some(layout)) = (mipmap_pipeline, mipmap_bind_group_layout) {
            if mip_level_count > 1 {
                Self::generate_mipmaps(
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

        for mip_level in 1..mip_level_count {
            let _mip_width = (width >> mip_level).max(1);
            let _mip_height = (height >> mip_level).max(1);

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

            let dest_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Mipmap Dest {}", mip_level)),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            });

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
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub material_bind_group: wgpu::BindGroup,
    pub diffuse_index: Option<usize>,
    pub normal_index: Option<usize>,
    pub mr_index: Option<usize>,
    pub diffuse_view: Rc<wgpu::TextureView>,
    pub normal_view: Rc<wgpu::TextureView>,
    pub mr_view: Rc<wgpu::TextureView>,
    pub sampler: Rc<wgpu::Sampler>,
    pub center: Vec3,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

pub struct Model {
    pub opaque_meshes: Vec<Mesh>,
    pub transparent_meshes: Vec<Mesh>,
    pub center: Vec3,
    pub extent: f32,
}

pub fn create_sphere_mesh(device: &wgpu::Device, queue: &wgpu::Queue, radius: f32, segments: u32, material_layout: &wgpu::BindGroupLayout) -> Mesh {
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
            positions.push([x * radius, y * radius, z * radius]);
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

pub fn compute_tangents(positions: &[[f32; 3]], normals: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]) -> Vec<[f32; 4]> {
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

pub fn load_model_from_bytes(
    device: &Rc<wgpu::Device>,
    queue: &Rc<wgpu::Queue>,
    bytes: &[u8],
    material_layout: &wgpu::BindGroupLayout,
    mipmap_pipeline_linear: &Rc<wgpu::RenderPipeline>,
    mipmap_pipeline_srgb: &Rc<wgpu::RenderPipeline>,
    mipmap_bind_group_layout: &Rc<wgpu::BindGroupLayout>,
    tx: Sender<AssetMessage>,
) -> Result<Model, String> {
    // HACK: Patch "extensionsRequired" to "extensionsOptional"
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
                web_sys::console::warn_1(&"Patched GLB: Bypassed extensionsRequired validation".into());
            }
        }
    }

    let gltf = gltf::Gltf::from_slice(&modified_bytes).map_err(|e| e.to_string())?;
    let document = gltf.document;
    let blob = gltf.blob.unwrap_or_default();
    
    let mut opaque_meshes = Vec::new();
    let mut transparent_meshes = Vec::new();
    let mut requested_textures = HashSet::new();

    let white_tex_srgb = Texture::single_pixel(device, queue, [255, 255, 255, 255], true);
    // let white_tex_linear = Texture::single_pixel(device, queue, [255, 255, 255, 255], false); // Unused currently
    let clay_mr_tex = Texture::single_pixel(device, queue, [255, 255, 0, 255], false);
    let normal_tex = Texture::single_pixel(device, queue, [128, 128, 255, 255], false);
    
    let white_view_srgb = Rc::new(white_tex_srgb.view);
    let clay_mr_view = Rc::new(clay_mr_tex.view);
    let normal_view = Rc::new(normal_tex.view);
    
    let sampler = Rc::new(device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Texture Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
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

                        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Mesh Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        
                        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Mesh Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                        let mat = primitive.material();
                        let pbr = mat.pbr_metallic_roughness();
                        
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
                                            let tx = tx.clone();
                                            let dev = device.clone();
                                            let q = queue.clone();
                                            let mipmap_pipeline_linear = mipmap_pipeline_linear.clone();
                                            let mipmap_pipeline_srgb = mipmap_pipeline_srgb.clone();
                                            let mipmap_layout = mipmap_bind_group_layout.clone();
                                            
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

                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: material_layout,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&white_view_srgb) },
                                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_view) },
                                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
                                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&clay_mr_view) },
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
    
    let mut model_min = vec3(f32::MAX, f32::MAX, f32::MAX);
    let mut model_max = vec3(f32::MIN, f32::MIN, f32::MIN);
    
    for mesh in opaque_meshes.iter().chain(transparent_meshes.iter()) {
        model_min = model_min.min(mesh.aabb_min);
        model_max = model_max.max(mesh.aabb_max);
    }
    
    let (center, extent) = if !opaque_meshes.is_empty() || !transparent_meshes.is_empty() {
        let c = (model_min + model_max) * 0.5;
        let ext = model_max - model_min;
        (c, ext.x.max(ext.y.max(ext.z)))
    } else {
        (Vec3::ZERO, 2.0)
    };
    
    Ok(Model { opaque_meshes, transparent_meshes, center, extent })
}
