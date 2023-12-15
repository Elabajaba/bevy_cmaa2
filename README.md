# bevy_cmaa

This is a WGSL rewrite of Intel's CMAA2 anti-aliasing algorithm. Quality wise it's expected to be similar to MSAA 2x, but with a similar cost to FXAA (though I expect at least the initial implementation here to end up slower than Intel's upstream implementation).

Compute shaders are required, so webgl2 and old GL aren't supported (WebGPU should probably work, but I won't know until I've gotten it running).

## TODO
 - [ ] Finish the bindings
 - [ ] Make the shader actually work
    - [ ] Either fix the `Pack_R11G11B10`/`Unpack_R11G11B10` work, or just use bevy's built in RGB9E5
 - Usage of `working_deferred_blend_item_list_heads` in the shader is currently wrong. Upstream uses a texture, but we can't here as WGSL doesn't support texture atomics.
 - [ ] License?

 ## Future work
  - [ ] Temporal CMAA2 using a Decima style 2 frame TAA?
  - [ ] Compare performance to FXAA and figure out if it's slower or not, and if it is can we do anything to make it faster? (I expect to initially use multiple swapchain copies for now since iirc bevy's main_texture can't be written to directly from a compute shader? And afaik bevy's current post processing stack relies on flip-flopping textures and doesn't let you write to one in place)
  - [ ] If it's as fast as FXAA could it maybe be used to anti-alias the disoccluded sections of the frame for TAA somehow? 


## Differences from Intel's upstream CMAA2
 - MSAA is unsupported.
 - This might end up using RGB9E5 textures instead of RG11B10 (/RG11B10E4 for non HDR textures?) so I don't have to write my own packing implementation. I'll probably also remove the HDR/LDR format switch as well then.
 - WGSL doesn't have textureatomics or RWByteAddressBuffer and idk if my workarounds even work, or what potential performance differences they might have.