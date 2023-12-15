use bevy::{
    asset::load_internal_asset,
    core_pipeline::{
        core_2d::{self, CORE_2D},
        core_3d::{self, CORE_3D},
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        globals::GlobalsBuffer,
        render_graph::{RenderGraphApp, ViewNodeRunner},
        render_resource::{
            AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingType, BufferBindingType, CachedComputePipelineId,
            ComputePipelineDescriptor, Extent3d, FilterMode, PipelineCache, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderDefVal, ShaderStages,
            SpecializedComputePipeline, SpecializedComputePipelines, StorageTextureAccess,
            TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
            TextureViewDimension,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::{CachedTexture, TextureCache},
        view::ViewUniforms,
        Extract, Render, RenderApp, RenderSet,
    },
};

use crate::cmaa_node::CmaaNode;

// Shaders:
// Main:
// EdgesColor2x2CS
// ProcessCandidatesCS
// DeferredColorApply2x2CS
// Helper shader for DispatchIndirect
// ComputeDispatchArgsCS
// Debug view shader
// DebugDrawEdgesCS

#[derive(Reflect, Eq, PartialEq, Hash, Clone, Copy)]
#[reflect(PartialEq, Hash)]
pub enum CmaaQualityPreset {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Reflect, Component, Clone, Eq, PartialEq, Hash, ExtractComponent)]
#[reflect(Component, Default)]
#[extract_component_filter(With<Camera>)]
pub struct CmaaSettings {
    pub texture_resolution: UVec2,
    pub preset: CmaaQualityPreset,
    pub enabled: bool,
    pub extra_sharpness: bool,
    pub sample_count: u8,
}

impl Default for CmaaSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            extra_sharpness: false,
            preset: CmaaQualityPreset::High,
            sample_count: 1,
            texture_resolution: UVec2::new(0, 0),
        }
    }
}

const CMAA_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(14405508006779432082128829898);

pub struct CmaaPlugin;

impl Plugin for CmaaPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, CMAA_SHADER_HANDLE, "cmaa.wgsl", Shader::from_wgsl);

        app.register_type::<CmaaSettings>();
        app.add_plugins(ExtractComponentPlugin::<CmaaSettings>::default());

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => {
                return;
            }
        };
        render_app
            .init_resource::<SpecializedComputePipelines<CmaaPipelineLayouts>>()
            .add_systems(ExtractSchedule, extract_cmaa_settings)
            .add_systems(
                Render,
                (
                    prepare_cmaa_pipelines.in_set(RenderSet::Prepare),
                    prepare_cmaa_textures.in_set(RenderSet::PrepareResources),
                    // prepare_cmaa_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<CmaaNode>>(CORE_3D, CmaaNode::NAME)
            .add_render_graph_edges(
                CORE_3D,
                &[
                    core_3d::graph::node::TONEMAPPING,
                    CmaaNode::NAME,
                    core_3d::graph::node::END_MAIN_PASS_POST_PROCESSING,
                ],
            )
            .add_render_graph_node::<ViewNodeRunner<CmaaNode>>(CORE_2D, CmaaNode::NAME)
            .add_render_graph_edges(
                CORE_3D,
                &[
                    core_2d::graph::node::TONEMAPPING,
                    CmaaNode::NAME,
                    core_2d::graph::node::END_MAIN_PASS_POST_PROCESSING,
                ],
            );
        // todo!();
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => {
                return;
            }
        };
        render_app.init_resource::<CmaaPipelineLayouts>();
    }
}

#[derive(Component)]
pub struct CmaaPipelines {
    pub(crate) edges_color_pipeline: CachedComputePipelineId,
    pub(crate) process_candidates_pipeline: CachedComputePipelineId,
    pub(crate) deferred_color_apply_pipeline: CachedComputePipelineId,
    /// Helper shader for DispatchIndirect
    pub(crate) compute_dispatch_args_pipeline: CachedComputePipelineId,
    /// Debug view shader
    pub(crate) debug_draw_edges_pipeline: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct CmaaPipelineLayouts {
    bind_group_layout: BindGroupLayout,

    point_clamp_sampler: Sampler,
    // EdgesColor2x2CS
    // ProcessCandidatesCS
    // DeferredColorApply2x2CS
    // ComputeDispatchArgsCS
    // DebugDrawEdgesCS
    // edges_color_bind_group_layout: BindGroupLayout,
    // process_candidates_bind_group_layout: BindGroupLayout,
    // deferred_color_apply_bind_group_layout: BindGroupLayout,
    // compute_dispatch_args_bind_group_layout: BindGroupLayout,
    // debug_draw_edges_bind_group_layout: BindGroupLayout,
}

impl FromWorld for CmaaPipelineLayouts {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let render_queue = render_world.resource::<RenderQueue>();
        let pipeline_cache = render_world.resource::<PipelineCache>();

        let point_clamp_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("cmaa_point_clamp_sampler"),
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..default()
        });

        let bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("cmaa_bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            // TODO: wrong format
                            format: TextureFormat::Rgba8UnormSrgb,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            // TODO: wrong format
                            format: TextureFormat::Rgba8UnormSrgb,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // var<storage, read_write> working_shape_candidates: array<u32>;
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // var<storage, read_write> working_deferred_blend_location_list: array<u32>;
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // var<storage, read_write> working_deferred_blend_item_list: array<vec2<u32>>; //
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // var<storage, read_write> working_deferred_blend_item_list_heads: array<atomic<u32>>;    // texture_storage_2d<u32, read_write>;
                    BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // var<storage, read_write> working_control_buffer: array<atomic<u32>>;
                    BindGroupLayoutEntry {
                        binding: 7,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // var<storage, read_write> working_execute_indirect_buffer: array<atomic<u32>>;
                    BindGroupLayoutEntry {
                        binding: 8,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // var inout_color_readonly : texture_2d<f32>; // input colour
                    BindGroupLayoutEntry {
                        binding: 9,
                        visibility: ShaderStages::COMPUTE,
                        // TODO: Might need to be a storage texture?
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true }, // TODO: maybe wrong
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        CmaaPipelineLayouts {
            // edges_color_pipeline,
            // process_candidates_pipeline,
            // deferred_color_apply_pipeline,
            // compute_dispatch_args_pipeline,
            // debug_draw_edges_pipeline,
            bind_group_layout,
            point_clamp_sampler,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct CmaaPipelineKey {
    cmaa_settings: CmaaSettings,
    pass: CmaaPass,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum CmaaPass {
    EdgesColor,
    ProcessCandidates,
    DeferredColorApply,
    ComputeDispatchArgs,
    DebugDrawEdges,
}

impl SpecializedComputePipeline for CmaaPipelineLayouts {
    type Key = CmaaPipelineKey;

    /// Add the shader_defs to the pipelines
    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![
            ShaderDefVal::UInt(
                "CMAA2_STATIC_QUALITY_PRESET".to_string(),
                key.cmaa_settings.preset as u32,
            ),
            ShaderDefVal::Bool(
                "CMAA2_EXTRA_SHARPNESS".to_string(),
                key.cmaa_settings.extra_sharpness,
            ),
            ShaderDefVal::UInt(
                "CMAA2_MSAA_SAMPLE_COUNT".to_string(),
                key.cmaa_settings.sample_count as u32,
            ),
            // TODO: Consider making these configurable options?
            // ShaderDefVal::UInt("CMAA2_EDGE_DETECTION_LUMA_PATH".to_string(), 1),
            // TODO: Figure out texture format stuff for these
            ShaderDefVal::UInt("CMAA2_UAV_STORE_TYPED".to_string(), 1),
            ShaderDefVal::UInt("CMAA2_UAV_STORE_CONVERT_TO_SRGB".to_string(), 1),
            ShaderDefVal::UInt("CMAA2_UAV_STORE_TYPED_UNORM_FLOAT".to_string(), 1), // 1 is UNORM, 0 is FLOAT
                                                                                    // TODO: This gets us a bit more precision for non-hdr colors, but it's not really necessary and it's unimplemented in the shader.
                                                                                    // See the Pack_R11G11B10_E4_FLOAT and Pack_R11G11B10_FLOAT functions in cmaa.wgsl
                                                                                    // ShaderDefVal::UInt("CMAA2_SUPPORT_HDR_COLOR_RANGE".to_string(), 1),
        ];

        // let edges_color_pipeline =
        //     pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        //         label: Some("cmaa_edges_color_pipeline".into()),
        //         layout: vec![bind_group_layout.clone()],
        //         push_constant_ranges: vec![],
        //         shader: CMAA_SHADER_HANDLE.clone(),
        //         shader_defs: Vec::new(),
        //         entry_point: std::borrow::Cow::Borrowed("EdgesColor2x2CS"),
        //     });
        // let process_candidates_pipeline =
        //     pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        //         label: Some("cmaa_process_candidates_pipeline".into()),
        //         layout: vec![bind_group_layout.clone()],
        //         push_constant_ranges: vec![],
        //         shader: CMAA_SHADER_HANDLE.clone(),
        //         shader_defs: Vec::new(),
        //         entry_point: std::borrow::Cow::Borrowed("ProcessCandidatesCS"),
        //     });
        // let deferred_color_apply_pipeline =
        //     pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        //         label: Some("cmaa_deferred_color_apply_pipeline".into()),
        //         layout: vec![bind_group_layout.clone()],
        //         push_constant_ranges: vec![],
        //         shader: CMAA_SHADER_HANDLE.clone(),
        //         shader_defs: Vec::new(),
        //         entry_point: std::borrow::Cow::Borrowed("DeferredColorApply2x2CS"),
        //     });
        // let compute_dispatch_args_pipeline =
        //     pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        //         label: Some("cmaa_compute_dispatch_args_pipeline".into()),
        //         layout: vec![bind_group_layout.clone()],
        //         push_constant_ranges: vec![],
        //         shader: CMAA_SHADER_HANDLE.clone(),
        //         shader_defs: Vec::new(),
        //         entry_point: std::borrow::Cow::Borrowed("ComputeDispatchArgsCS"),
        //     });
        // let debug_draw_edges_pipeline =
        //     pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        //         label: Some("cmaa_debug_draw_edges_pipeline".into()),
        //         layout: vec![bind_group_layout.clone()],
        //         push_constant_ranges: vec![],
        //         shader: CMAA_SHADER_HANDLE.clone(),
        //         shader_defs: Vec::new(),
        //         entry_point: std::borrow::Cow::Borrowed("DebugDrawEdgesCS"),
        //     });

        match key.pass {
            CmaaPass::EdgesColor => ComputePipelineDescriptor {
                label: Some("cmaa_edges_color_pipeline".into()),
                layout: vec![self.bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: CMAA_SHADER_HANDLE.clone(),
                shader_defs: Vec::new(),
                entry_point: std::borrow::Cow::Borrowed("EdgesColor2x2CS"),
            },
            CmaaPass::ProcessCandidates => ComputePipelineDescriptor {
                label: Some("cmaa_process_candidates_pipeline".into()),
                layout: vec![self.bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: CMAA_SHADER_HANDLE.clone(),
                shader_defs: Vec::new(),
                entry_point: std::borrow::Cow::Borrowed("ProcessCandidatesCS"),
            },
            CmaaPass::DeferredColorApply => ComputePipelineDescriptor {
                label: Some("cmaa_deferred_color_apply_pipeline".into()),
                layout: vec![self.bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: CMAA_SHADER_HANDLE.clone(),
                shader_defs: Vec::new(),
                entry_point: std::borrow::Cow::Borrowed("DeferredColorApply2x2CS"),
            },
            CmaaPass::ComputeDispatchArgs => ComputePipelineDescriptor {
                label: Some("cmaa_compute_dispatch_args_pipeline".into()),
                layout: vec![self.bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: CMAA_SHADER_HANDLE.clone(),
                shader_defs: Vec::new(),
                entry_point: std::borrow::Cow::Borrowed("ComputeDispatchArgsCS"),
            },
            CmaaPass::DebugDrawEdges => ComputePipelineDescriptor {
                label: Some("cmaa_debug_draw_edges_pipeline".into()),
                layout: vec![self.bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: CMAA_SHADER_HANDLE.clone(),
                shader_defs: Vec::new(),
                entry_point: std::borrow::Cow::Borrowed("DebugDrawEdgesCS"),
            },
        };
        // TODO: Do all the pipelines need to be specialized?
        todo!()
    }
}

#[derive(Component)]
pub struct CmaaTextures {
    // inout_color_writeonly: todo!(),
    // inout_color_readonly: todo!(),
    pub(crate) inout_color: CachedTexture,
    pub(crate) working_edges: CachedTexture,
}

fn prepare_cmaa_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera), With<CmaaSettings>>,
) {
    for (entity, camera) in &views {
        let Some(physical_viewport_size) = camera.physical_viewport_size else {
            continue;
        };
        let size = Extent3d {
            width: physical_viewport_size.x,
            height: physical_viewport_size.y,
            depth_or_array_layers: 1,
        };

        let inout_color = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cmaa_inout_color"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        let working_edges = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cmaa_working_edges"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(CmaaTextures {
            inout_color,
            working_edges,
        });
    }
}

fn prepare_cmaa_pipelines(
    mut commands: Commands,
    mut pipelines: ResMut<SpecializedComputePipelines<CmaaPipelineLayouts>>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CmaaPipelineLayouts>,
    views: Query<(Entity, &CmaaSettings), With<CmaaTextures>>,
) {
    // Borrowed from https://github.com/JMS55/bevy/blob/1268a3f2b8fa9840aab9ec3ca89e6dd1d9e2b55d/crates/bevy_pbr/src/solari/global_illumination/pipelines.rs#L175
    let mut create_pipeline = |key| pipelines.specialize(&pipeline_cache, &pipeline, key);

    for (entity, settings) in &views {
        commands.entity(entity).insert(CmaaPipelines {
            edges_color_pipeline: create_pipeline(CmaaPipelineKey {
                cmaa_settings: settings.clone(),
                pass: CmaaPass::EdgesColor,
            }),
            process_candidates_pipeline: create_pipeline(CmaaPipelineKey {
                cmaa_settings: settings.clone(),
                pass: CmaaPass::ProcessCandidates,
            }),
            deferred_color_apply_pipeline: create_pipeline(CmaaPipelineKey {
                cmaa_settings: settings.clone(),
                pass: CmaaPass::DeferredColorApply,
            }),
            compute_dispatch_args_pipeline: create_pipeline(CmaaPipelineKey {
                cmaa_settings: settings.clone(),
                pass: CmaaPass::ComputeDispatchArgs,
            }),
            debug_draw_edges_pipeline: create_pipeline(CmaaPipelineKey {
                cmaa_settings: settings.clone(),
                pass: CmaaPass::DebugDrawEdges,
            }),
        });
    }
}

// Have to do this in Node::run because it uses post_process_write
// fn prepare_cmaa_bind_groups(
//     mut commands: Commands,
//     render_device: Res<RenderDevice>,
//     view_uniforms: Res<ViewUniforms>,
//     global_uniforms: Res<GlobalsBuffer>,
//     views: Query<(Entity, &CmaaTextures)>,
//     mut pipelines: ResMut<SpecializedComputePipelines<CmaaPipelines>>,
//     mut cmaa_pipelines: ResMut<CmaaPipelines>,
// ) {
//     let (Some(view_uniforms), Some(global_uniforms)) = (
//         view_uniforms.uniforms.binding(),
//         global_uniforms.buffer.binding(),
//     ) else {
//         return;
//     };

//     for (entity, cmaa_textures) in &views {
//         let bind_group = render_device.create_bind_group(
//             "cmaa_bind_group",
//             &cmaa_pipelines.bind_group_layout,
//             &BindGroupEntries::sequential(()),
//         );
//     }
// }

// @group(0) @binding(0)
// var gather_point_clamp_sampler: sampler;
// #ifdef CMAA2_UAV_STORE_TYPED
// #ifdef CMAA2_UAV_STORE_TYPED_UNORM_FLOAT
// @group(0) @binding(1)
// var inout_color_writeonly: texture_storage_2d<f32, write>; // final output colour: TODO ifdef support formats?
// #else
// @group(0) @binding(1)
// var inout_color_writeonly: texture_storage_2d<vec4<f32>, write>; // final output colour: TODO ifdef support formats?
// #endif
// #else
// @group(0) @binding(1)
// var inout_color_writeonly: texture_storage_2d<u32, write>; // final output colour: TODO ifdef support formats?
// #endif
// #ifdef CMAA2_EDGE_UNORM
// @group(0) @binding(2)
// var working_edges: texture_storage_2d<f32, write>; // output edges (only used in the first pass)
// #else
// @group(0) @binding(2)
// var working_edges: texture_storage_2d<u32, write>; // output edges (only used in the first pass)
// #endif
// @group(0) @binding(3)
// var<storage, read_write> working_shape_candidates: array<u32>;
// @group(0) @binding(4)
// var<storage, read_write> working_deferred_blend_location_list: array<u32>;
// @group(0) @binding(5)
// var<storage, read_write> working_deferred_blend_item_list: array<vec2<u32>>; //
// @group(0) @binding(6)
// var<storage, read_write> working_deferred_blend_item_list_heads: array<atomic<u32>>;    // texture_storage_2d<u32, read_write>;
// @group(0) @binding(7)
// var<storage, read_write> working_control_buffer: array<atomic<u32>>;
// @group(0) @binding(8)
// var<storage, read_write> working_execute_indirect_buffer: array<atomic<u32>>;
// @group(0) @binding(9)
// var inout_color_readonly : texture_2d<f32>; // input colour

fn extract_cmaa_settings(
    mut commands: Commands,
    cameras: Extract<Query<(Entity, &Camera, &CmaaSettings)>>,
) {
    for (camera_entity, camera, cmaa_settings) in cameras.iter() {
        if camera.is_active {
            commands
                .get_or_spawn(camera_entity)
                .insert(cmaa_settings.clone());
        }
    }
}
