use std::sync::Mutex;

use bevy::{
    ecs::query::QueryItem,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_resource::{PipelineCache, ComputePassDescriptor, BufferId, TextureViewId, BindGroup},
        renderer::RenderContext,
        view::ViewTarget,
    },
};

use crate::cmaa::{CmaaPipelineLayouts, CmaaPipelines, CmaaSettings};

// The post process node used for the render graph
#[derive(Default)]
pub struct CmaaNode {
    cached_bind_group: Mutex<Option<(BufferId, TextureViewId, BindGroup)>>,
};
impl CmaaNode {
    pub const NAME: &'static str = "cmaa";
}

impl ViewNode for CmaaNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        // &'static CmaaPipelines,
        &'static CmaaSettings,
        &'static CmaaPipelines,
    );

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, view_target, cmaa_settings, cmaa_pipelines): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        // let pipelines = world.resource::<CmaaPipelineLayouts>();

        if !cmaa_settings.enabled {
            return Ok(());
        }



        let (
            Some(camera_size),
            Some(edges_color_pipeline),
            Some(process_candidates_pipeline),
            Some(deferred_color_apply_pipeline),
            Some(compute_dispatch_args_pipeline),
            Some(debug_draw_edges_pipeline),
            // edges_color_pipeline: CachedComputePipelineId,
            // process_candidates_pipeline: CachedComputePipelineId,
            // deferred_color_apply_pipeline: CachedComputePipelineId,
            // compute_dispatch_args_pipeline: CachedComputePipelineId,
            // debug_draw_edges_pipeline: CachedComputePipelineId,
        ) = (
            // TODO: Should this be the camera viewport size, or the viewtarget size?
            camera.physical_viewport_size,
            pipeline_cache.get_compute_pipeline(cmaa_pipelines.edges_color_pipeline),
            pipeline_cache.get_compute_pipeline(cmaa_pipelines.process_candidates_pipeline),
            pipeline_cache.get_compute_pipeline(cmaa_pipelines.deferred_color_apply_pipeline),
            pipeline_cache.get_compute_pipeline(cmaa_pipelines.compute_dispatch_args_pipeline),
            pipeline_cache.get_compute_pipeline(cmaa_pipelines.debug_draw_edges_pipeline),
        )
        else {
            return Ok(());
        };

        // TODO: The shader can write in place, so how do we do that to avoid having to copy these?
        let target = view_target.post_process_write();
        let source = target.source;
        let destination = target.destination;

        


        let command_encoder = render_context.command_encoder();
        let mut cmaa_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cmaa2"),
        });
        // TODO: What push constants do we need?
        cmaa_pass.push_debug_group("label");
        cmaa_pass.set_pipeline(&edges_color_pipeline);
        cmaa_pass.dispatch_workgroups_indirect(indirect_buffer, indirect_offset)
        Ok(())
    }
}
