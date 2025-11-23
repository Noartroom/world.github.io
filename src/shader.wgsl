struct AudioUniform {
    intensity: f32,
    balance: f32,
    _pad1: f32,
    _pad2: f32,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct LightUniform {
    position: vec4<f32>,
    color: vec4<f32>,
    ambient_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> audio: AudioUniform;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> light: LightUniform;

// Material Bindings (Group 3)
@group(3) @binding(0) var t_diffuse: texture_2d<f32>;
@group(3) @binding(1) var s_diffuse: sampler;
@group(3) @binding(2) var t_normal: texture_2d<f32>;
@group(3) @binding(3) var s_normal: sampler;
@group(3) @binding(4) var t_mr: texture_2d<f32>; // Metallic-Roughness
@group(3) @binding(5) var s_mr: sampler;

// --- SKY RENDERER (Procedural Sky) ---
// Renders a full-screen triangle

struct SkyOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_dir: vec3<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) in_vertex_index: u32) -> SkyOutput {
    var out: SkyOutput;
    // Robust Fullscreen Triangle (covering [-1, 1] range)
    let uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    let x = uv.x * 2.0 - 1.0;
    let y = uv.y * 2.0 - 1.0; // Inverted Y might be needed depending on unproj? Let's try standard first.

    // Position at depth = 1.0 (far plane)
    // Note: For the background to be behind everything, we use z = 1.0 (if depth test is Less/Equal)
    // BUT we often need to force it to max depth. In WebGPU standard depth [0, 1].
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);
    
    // Unprojection
    // We pass the clip position to unproject.
    // Depending on how the projection matrix was built (GL vs Direct3D style), Y might need flip.
    // Standard WGPU projection (perspective_rh) usually results in Y-up.
    // Let's try with standard x,y. If sky is upside down, we flip Y.
    let clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    
    let unproj = camera.inv_view_proj * clip_pos;
    out.view_dir = normalize(unproj.xyz / unproj.w - camera.camera_pos.xyz);
    
    return out;
}

@fragment
fn fs_sky(in: SkyOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.view_dir);
    
    // Gradient Sky based on Y (up)
    let t = 0.5 * (dir.y + 1.0);
    
    // Simple Day/Night mix based on light intensity or audio?
    // Let's use Audio to modulate sky color!
    let base_color = mix(vec3<f32>(0.02, 0.02, 0.05), vec3<f32>(0.1, 0.2, 0.4), t); // Dark Space -> Blueish
    let beat_color = vec3<f32>(0.5, 0.2, 0.8) * audio.intensity;
    
    // Stars or noise could be added here
    
    return vec4<f32>(base_color + beat_color * 0.2, 1.0);
}

// --- GRID RENDERER (Legacy / Optional) ---
// Using same group layout: 0=Audio, 1=Camera, 2=Light

struct GridVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

const GRID_SIZE: u32 = 100u;
const SPACING: f32 = 0.5;

@vertex
fn vs_grid(@builtin(vertex_index) in_vertex_index: u32) -> GridVertexOutput {
    var out: GridVertexOutput;

    let quad_index = in_vertex_index / 6u;
    let local_index = in_vertex_index % 6u;

    let grid_x = f32(quad_index % GRID_SIZE);
    let grid_y = f32(quad_index / GRID_SIZE);

    var offset = vec2<f32>(0.0, 0.0);
    if (local_index == 1u || local_index == 3u) { offset = vec2<f32>(1.0, 0.0); }
    else if (local_index == 2u || local_index == 5u) { offset = vec2<f32>(0.0, 1.0); }
    else if (local_index == 4u) { offset = vec2<f32>(1.0, 1.0); }

    let x_raw = grid_x + offset.x;
    let y_raw = grid_y + offset.y;
    
    let center_offset = f32(GRID_SIZE) * 0.5;
    var x = (x_raw - center_offset) * SPACING;
    var z_grid = (y_raw - center_offset) * SPACING; 
    
    // Audio Deformation
    let dist = sqrt(x*x + z_grid*z_grid);
    let x_norm = x / (f32(GRID_SIZE) * SPACING * 0.5);
    let directional_weight = 1.0 + (x_norm * audio.balance);
    let wave = sin(dist * 2.0 - audio.intensity * 10.0);
    
    var y_deformation = 0.0;
    y_deformation += wave * audio.intensity * 1.0 * directional_weight;
    
    let world_pos = vec4<f32>(x, -2.0 + y_deformation, z_grid, 1.0);
    out.clip_position = camera.view_proj * world_pos;

    // Grid Color Logic
    let line_width = 0.02;
    let is_grid_x = step(1.0 - line_width, fract(x_raw));
    let is_grid_y = step(1.0 - line_width, fract(y_raw));
    let is_grid = max(is_grid_x, is_grid_y);

    var color = vec3<f32>(0.01, 0.01, 0.02); 
    
    if (is_grid > 0.5) {
        let alpha = 1.0 - clamp(dist / (f32(GRID_SIZE) * SPACING * 0.5), 0.0, 1.0);
        color = vec3<f32>(0.0, 0.5, 1.0) * (0.2 + audio.intensity) * alpha;
    }
    
    out.color = color;
    return out;
}

@fragment
fn fs_grid(in: GridVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

// --- MODEL RENDERER (Foreground) ---

const PI: f32 = 3.14159265359;

// Helper Functions for PBR
fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let num = a2;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return num / (PI * denom * denom);
}
fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = geometrySchlickGGX(NdotV, roughness);
    let ggx2 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}
fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

struct ModelInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct ModelOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent_view: vec3<f32>,
    @location(4) bitangent_view: vec3<f32>,
    @location(5) normal_view: vec3<f32>,
};

@vertex
fn vs_model(model: ModelInput) -> ModelOutput {
    var out: ModelOutput;
    
    // Vertices are already pre-transformed to World Space on the CPU
    let world_pos = vec4<f32>(model.position, 1.0);
    out.world_pos = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    out.uv = model.uv;
    
    // Normals are also pre-transformed
    out.normal = model.normal;
    out.normal_view = model.normal;
    
    // Tangent reconstruction (pre-transformed)
    out.tangent_view = model.tangent.xyz;
    let N = normalize(model.normal);
    let T = normalize(model.tangent.xyz);
    out.bitangent_view = cross(N, T) * model.tangent.w;
    
    return out;
}
@fragment
fn fs_model(in: ModelOutput) -> @location(0) vec4<f32> {
    // Sample Textures
    let albedo = textureSample(t_diffuse, s_diffuse, in.uv).rgb;
    let mr = textureSample(t_mr, s_mr, in.uv);
    let metallic = mr.b;
    let roughness = mr.g;
    let occlusion = mr.r;

    // Normal Mapping
    let tnorm = textureSample(t_normal, s_normal, in.uv).xyz * 2.0 - 1.0;
    let T = normalize(in.tangent_view);
    let B = normalize(in.bitangent_view);
    let N_geom = normalize(in.normal_view);
    let TBN = mat3x3<f32>(T, B, N_geom);
    let N = normalize(TBN * tnorm);

    let V = normalize(camera.camera_pos.xyz - in.world_pos);
    
    var F0 = vec3<f32>(0.04); 
    F0 = mix(F0, albedo, metallic);

    // Dynamic Light from Uniform (Point Light with Distance Attenuation)
    let light_pos = light.position.xyz;
    let light_dir = light_pos - in.world_pos;
    let light_distance = length(light_dir);
    let L = normalize(light_dir);
    let H = normalize(V + L);
    
    // Point light attenuation (inverse square law with minimum distance)
    // Use smoother attenuation for better visibility
    let min_distance = 0.5;
    let attenuation_distance = max(light_distance, min_distance);
    // Inverse square law with smoother falloff
    let attenuation = 1.0 / (1.0 + 0.1 * attenuation_distance + 0.01 * attenuation_distance * attenuation_distance);
    // Boost intensity for better visibility
    let intensity_multiplier = 3.0;
    
    // Lighting Calculation
    let NDF = distributionGGX(N, H, roughness);   
    let G = geometrySmith(N, V, L, roughness);      
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
       
    let num = NDF * G * F; 
    let den = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = num / den;
    
    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD = kD * (1.0 - metallic);	  
    
    let NdotL = max(dot(N, L), 0.0);
    
    // Light Color & Intensity (from Uniforms only - no audio influence)
    // Apply distance attenuation for point light (omnidirectional)
    let lightColor = light.color.rgb;
    let radiance = lightColor * attenuation * intensity_multiplier; 
        
    let Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Ambient / Emission
    let ambient = light.ambient_color.rgb * albedo * occlusion;
    // Add emission for blob visibility (if material is highly emissive/metallic)
    // This makes the blob glow even when not directly lit
    let emission = albedo * metallic * 0.5; // Emissive glow based on material properties 

    let color = ambient + Lo + emission;
    
    // Improved Tone Mapping (ACES approximation - better than simple Reinhard)
    // This provides more natural color reproduction and better highlight handling
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let mapped = clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Note: Gamma correction is handled automatically by the GPU when outputting to sRGB format
    // If the surface format is sRGB (which it should be), the GPU will apply gamma correction
    // So we output linear color here and let the GPU handle the conversion
    
    return vec4<f32>(mapped, 1.0);
}

// --- MIPMAP GENERATION SHADER ---
// Used to generate mip levels by downsampling from the previous level

@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_source: sampler;

struct MipmapVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_mipmap(@builtin(vertex_index) in_vertex_index: u32) -> MipmapVertexOutput {
    var out: MipmapVertexOutput;
    // Fullscreen triangle
    let uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_mipmap(in: MipmapVertexOutput) -> @location(0) vec4<f32> {
    // Box filter: sample 4 texels and average them
    // This provides good quality for mipmap generation
    let texel_size = 1.0 / vec2<f32>(textureDimensions(t_source));
    
    let a = textureSample(t_source, s_source, in.uv + vec2<f32>(-0.5, -0.5) * texel_size);
    let b = textureSample(t_source, s_source, in.uv + vec2<f32>(0.5, -0.5) * texel_size);
    let c = textureSample(t_source, s_source, in.uv + vec2<f32>(-0.5, 0.5) * texel_size);
    let d = textureSample(t_source, s_source, in.uv + vec2<f32>(0.5, 0.5) * texel_size);
    
    return (a + b + c + d) * 0.25;
}