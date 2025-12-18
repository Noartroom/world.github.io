use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct LightUniform {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub sky_color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct BlobUniform {
    pub position: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SceneUniform {
    pub camera: CameraUniform,
    pub light: LightUniform,
    pub blob: BlobUniform,
    pub time: f32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
    pub tangent: [f32; 4],
}

impl ModelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x4
    ];
    
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}
