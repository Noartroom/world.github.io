use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[cfg(feature = "console_error_panic_hook")]
use std::panic;

#[wasm_bindgen]
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl State {
    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

#[wasm_bindgen]
impl State {
    #[wasm_bindgen(js_name = "setTheme")]
    pub fn set_theme(&mut self, is_dark: bool) {
        web_sys::console::log_1(&format!("Theme updated to: {}", if is_dark { "Dark" } else { "Light" }).into());
        // Future: update clear color or shader uniforms based on theme
    }

    #[wasm_bindgen(js_name = "loadModelFromBytes")]
    pub fn load_model_from_bytes(&mut self, bytes: &[u8]) {
        web_sys::console::log_1(&format!("Model bytes received: {} bytes", bytes.len()).into());
        // Future: Parse GLTF/GLB here
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            
            let (depth_texture, depth_view) = State::create_depth_texture(&self.device, &self.config);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    pub fn update_camera(&mut self, _dx: f32, _dy: f32, _zoom: f32) {
        // web_sys::console::log_1(&format!("Camera update: dx={}, dy={}, zoom={}", dx, dy, zoom).into());
        // Future: Update camera controller state
    }

    pub fn update(&mut self, _cursor_x: f32, _cursor_y: f32, _width: f32, _height: f32) {
        // Future: Update lighting uniforms based on cursor
    }

    pub fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => {
                self.resize(self.config.width, self.config.height);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                web_sys::console::error_1(&"WGPU OutOfMemory".into());
                return;
            }
            Err(e) => {
                web_sys::console::warn_1(&format!("{:?}", e).into());
                return;
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            // 1. Depth Pre-Pass (Z-Prepass)
            // Write to depth buffer only. No fragment shader (implied by pipeline layout in future).
            // This allows early-z rejection in the color pass.
            let _depth_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth Pre-Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        {
            // 2. Color Pass
            // Render the scene using the depth buffer for comparison.
            // Only pixels that match the depth buffer (Equal) will be shaded.
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Color Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15, // Dark blue-ish background
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Use the depth buffer from the pre-pass
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

#[wasm_bindgen(js_name = "startRenderer")]
pub async fn start_renderer(canvas: HtmlCanvasElement) -> Result<State, JsValue> {
    #[cfg(feature = "console_error_panic_hook")]
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    web_sys::console::log_1(&"Initializing WGPU Renderer...".into());

    let instance = wgpu::Instance::default();
    
    // Create surface from canvas
    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {}", e)))?;

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }).await.ok_or_else(|| JsValue::from_str("Failed to find an appropriate adapter"))?;

    // Request device and queue
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            // WebGL2 defaults for compatibility
            required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits()),
            label: None,
        },
        None,
    ).await.map_err(|e| JsValue::from_str(&format!("Failed to create device: {}", e)))?;

    let surface_caps = surface.get_capabilities(&adapter);
    // Select sRGB format if available
    let surface_format = surface_caps.formats.iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: 100, // Initial dummy size, will be resized immediately
        height: 100,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    
    surface.configure(&device, &config);

    let (depth_texture, depth_view) = State::create_depth_texture(&device, &config);

    web_sys::console::log_1(&"WGPU Renderer initialized successfully (Depth Pre-Pass Enabled)".into());

    Ok(State {
        surface,
        device,
        queue,
        config,
        depth_texture,
        depth_view,
    })
}