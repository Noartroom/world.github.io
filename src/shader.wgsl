struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct LightUniform {
    position: vec4<f32>,
    color: vec4<f32>,
    sky_color: vec4<f32>,
};

// NEW: Uniform for the Blob position
struct BlobUniform {
    position: vec4<f32>,
    color: vec4<f32>,
};

// Consolidated scene uniforms
struct SceneUniform {
    camera: CameraUniform,
    light: LightUniform,
    blob: BlobUniform,
};

@group(0) @binding(0) var<uniform> scene: SceneUniform;

// Material Bindings (Group 1)
@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(1) @binding(3) var s_normal: sampler;
@group(1) @binding(4) var t_mr: texture_2d<f32>; // Metallic-Roughness
@group(1) @binding(5) var s_mr: sampler;


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
    
    // STANDARD MODEL: No offset
    let world_pos = vec4<f32>(model.position, 1.0);
    out.world_pos = world_pos.xyz;
    out.clip_position = scene.camera.view_proj * world_pos;
    out.uv = model.uv;
    
    out.normal = model.normal;
    out.normal_view = model.normal;
    
    out.tangent_view = model.tangent.xyz;
    let N = normalize(model.normal);
    let T = normalize(model.tangent.xyz);
    out.bitangent_view = cross(N, T) * model.tangent.w;
    
    return out;
}

// NEW: Vertex shader specifically for the Blob
@vertex
fn vs_blob(model: ModelInput) -> ModelOutput {
    var out: ModelOutput;
    
    // Apply Uniform Offset here (Cheap!)
    let world_pos = vec4<f32>(model.position + scene.blob.position.xyz, 1.0);
    
    out.world_pos = world_pos.xyz;
    out.clip_position = scene.camera.view_proj * world_pos;
    out.uv = model.uv;
    
    out.normal = model.normal;
    out.normal_view = model.normal;
    
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

    let V = normalize(scene.camera.camera_pos.xyz - in.world_pos);
    
    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);

    // Dynamic Light from Uniform (Point Light with Distance Attenuation)
    let light_pos = scene.light.position.xyz;
    let light_dir = light_pos - in.world_pos;
    let light_distance = length(light_dir);
    let L = normalize(light_dir);
    let H = normalize(V + L);
    
    // Point light attenuation (inverse square law with minimum distance)
    // Use smoother, wider attenuation for better visibility and wider light cone
    let min_distance = 1.5; // Increased from 0.5 - wider light cone
    let attenuation_distance = max(light_distance, min_distance);
    // Inverse square law with much smoother falloff for wider light cone
    // Reduced coefficients make light fall off more slowly, creating a wider area of effect
    let attenuation = 1.0 / (1.0 + 0.05 * attenuation_distance + 0.005 * attenuation_distance * attenuation_distance);
    // Intensity multiplier for visible light
    let intensity_multiplier = 4.0; // Reduced from 5.0
    
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
    let lightColor = scene.light.color.rgb;
    let radiance = lightColor * attenuation * intensity_multiplier;
        
    let Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Ambient (Hemispheric) with Soft Ground Shadows
    // Blend between ground and sky color based on normal Y
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let w_linear = 0.5 * (dot(N, up) + 1.0);
    
    // "Dreamworld" Gradient Tuning:
    // Use a slightly wider smoothstep to mimic the CSS gradient's softness,
    // but keep the bottom darker for grounding.
    // 0.2 to 0.8 gives a smooth transition across the equator.
    let w = smoothstep(0.2, 0.8, w_linear);
    
    // Mix with a bias: darken the ground color significantly (10% of sky) to simulate soft occlusion shadows at the base
    let ambient_light = mix(scene.light.sky_color.rgb * 0.1, scene.light.sky_color.rgb, w);
    
    let ambient = ambient_light * albedo * occlusion;
    
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