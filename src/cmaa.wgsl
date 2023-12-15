///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Intel Corporation
//
// Licensed under the Apache License, Version 2.0 ( the "License" );
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conservative Morphological Anti-Aliasing, version: 2.3
//
// Author(s):       Filip Strugar (filip.strugar@intel.com)
//
// More info:       https://github.com/GameTechDev/CMAA2
//
// Please see https://github.com/GameTechDev/CMAA2/README.md for additional information and a basic integration guide.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Constants that C++/API side needs to know!
#define CMAA_PACK_SINGLE_SAMPLE_EDGE_TO_HALF_WIDTH 1   // adds more ALU but reduces memory use for edges by half by packing two 4 bit edge info into one R8_UINT texel - helps on all HW except at really low res
#define CMAA2_CS_INPUT_KERNEL_SIZE_X 16
#define CMAA2_CS_INPUT_KERNEL_SIZE_Y 16

// The rest below is shader only code

// If the color buffer range is bigger than [0, 1] then use this, otherwise don't (and gain some precision - see https://bartwronski.com/2017/04/02/small-float-formats-r11g11b10f-precision/)
#ifndef CMAA2_SUPPORT_HDR_COLOR_RANGE
#define CMAA2_SUPPORT_HDR_COLOR_RANGE 0
#endif

// // 0 is full color-based edge detection, 1 and 2 are idential log luma based, with the difference bing that 1 loads color and computes log luma in-place (less efficient) while 2 loads precomputed log luma from a separate R8_UNORM texture (more efficient).
// // Luma-based edge detection has a slightly lower quality but better performance so use it as a default; providing luma as a separate texture (or .a channel of the main one) will improve performance.
// // See RGBToLumaForEdges for luma conversions in non-HDR and HDR versions.
#ifndef CMAA2_EDGE_DETECTION_LUMA_PATH
#define CMAA2_EDGE_DETECTION_LUMA_PATH 1
#endif

// // for CMAA2+MSAA support
#ifndef CMAA_MSAA_SAMPLE_COUNT
#define CMAA_MSAA_SAMPLE_COUNT 1
#endif

#define CMAA2_CS_OUTPUT_KERNEL_SIZE_X (CMAA2_CS_INPUT_KERNEL_SIZE_X-2)
#define CMAA2_CS_OUTPUT_KERNEL_SIZE_Y (CMAA2_CS_INPUT_KERNEL_SIZE_Y-2)
#define CMAA2_PROCESS_CANDIDATES_NUM_THREADS 128
#define CMAA2_DEFERRED_APPLY_NUM_THREADS 32

// // Optimization paths
#define CMAA2_DEFERRED_APPLY_THREADGROUP_SWAP 1   // 1 seems to be better or same on all HW
#define CMAA2_COLLECT_EXPAND_BLEND_ITEMS 1   // this reschedules final part of work in the ProcessCandidatesCS (where the sampling and blending takes place) from few to all threads to increase hardware thread occupancy
// #ifndef CMAA2_USE_HALF_FLOAT_PRECISION                  
// #define CMAA2_USE_HALF_FLOAT_PRECISION 0   // use half precision by default? (not on by default due to driver issues on various different hardware, but let external code decide to define if needed)
// #endif

#ifndef CMAA2_UAV_STORE_TYPED
// #error Warning - make sure correct value is set according to D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE caps for the color UAV format used in g_inout_colorWriteonly
#define CMAA2_UAV_STORE_TYPED 1   // use defaults that match the most common scenario: DXGI_FORMAT_R8G8B8A8_UNORM as UAV on a DXGI_FORMAT_R8G8B8A8_UNORM_SRGB resource (no typed stores for sRGB so we have to manually convert)
#endif

#ifndef CMAA2_UAV_STORE_CONVERT_TO_SRGB
// #error Warning - make sure correct value is set according to whether manual linear->sRGB color conversion is needed when writing color output to g_inout_colorWriteonly
#define CMAA2_UAV_STORE_CONVERT_TO_SRGB 1   // use defaults that match the most common scenario: DXGI_FORMAT_R8G8B8A8_UNORM as UAV on a DXGI_FORMAT_R8G8B8A8_UNORM_SRGB resource (no typed stores for sRGB so we have to manually convert)
#endif

#ifndef CMAA2_UAV_STORE_TYPED_UNORM_FLOAT
// // #error Warning - make sure correct value is set according to the color UAV format used in g_inout_colorWriteonly
#define CMAA2_UAV_STORE_TYPED_UNORM_FLOAT 1   // for typed UAV stores: set to 1 for all _UNORM formats and to 0 for _FLOAT formats
#endif

// #if CMAA2_UAV_STORE_TYPED
//     #ifndef CMAA2_UAV_STORE_TYPED_UNORM_FLOAT
//         // #error When CMAA2_UAV_STORE_TYPED is set to 1, CMAA2_UAV_STORE_TYPED_UNORM_FLOAT must be set 1 if the color UAV is not a _FLOAT format or 0 if it is.
//     #endif
// #else
//     #ifndef CMAA2_UAV_STORE_UNTYPED_FORMAT
//         // #error Error - untyped format required (see FinalUAVStore function for the list)
//     #endif
// #endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// VARIOUS QUALITY SETTINGS
//
// Longest line search distance; must be even number; for high perf low quality start from ~32 - the bigger the number, 
// the nicer the gradients but more costly. Max supported is 128!
const c_maxLineLength: u32 = 86u;
// 
#ifndef CMAA2_EXTRA_SHARPNESS
    #define CMAA2_EXTRA_SHARPNESS                   0     // Set to 1 to preserve even more text and shape clarity at the expense of less AA
#endif
//
// It makes sense to slightly drop edge detection thresholds with increase in MSAA sample count, as with the higher
// MSAA level the overall impact of CMAA2 alone is reduced but the cost increases.
#define CMAA2_SCALE_QUALITY_WITH_MSAA               0
//
// 
#ifndef CMAA2_STATIC_QUALITY_PRESET
    #define CMAA2_STATIC_QUALITY_PRESET 2  // 0 - LOW, 1 - MEDIUM, 2 - HIGH, 3 - ULTRA
#endif
// presets (for HDR color buffer maybe use higher values)
#if CMAA2_STATIC_QUALITY_PRESET == 0   // LOW
    #define g_CMAA2_EdgeThreshold 0.15
#else if CMAA2_STATIC_QUALITY_PRESET == 1 // MEDIUM
    #define g_CMAA2_EdgeThreshold 0.10
#else if CMAA2_STATIC_QUALITY_PRESET == 2 // HIGH (default)
    #define g_CMAA2_EdgeThreshold 0.07
#else if CMAA2_STATIC_QUALITY_PRESET == 3 // ULTRA
    #define g_CMAA2_EdgeThreshold 0.05
// #else
    // #error CMAA2_STATIC_QUALITY_PRESET not set?
#endif
// 
#ifdef CMAA2_EXTRA_SHARPNESS
#define g_CMAA2_LocalContrastAdaptationAmount 0.15
#define g_CMAA2_SimpleShapeBlurinessAmount 0.07
#else
#define g_CMAA2_LocalContrastAdaptationAmount 0.10
#define g_CMAA2_SimpleShapeBlurinessAmount 0.10
#endif
// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// #if CMAA_MSAA_SAMPLE_COUNT > 1
// #define CMAA_MSAA_USE_COMPLEXITY_MASK 1
// #endif

// #if CMAA2_EDGE_DETECTION_LUMA_PATH == 2 || CMAA2_EDGE_DETECTION_LUMA_PATH == 3 || CMAA_MSAA_USE_COMPLEXITY_MASK
// SamplerState                    g_gather_point_clamp_Sampler        : register( s0 );       // there's also a slightly less efficient codepath that avoids Gather for easier porting
// #endif

// // Is the output UAV format R32_UINT for manual shader packing, or a supported UAV store format?
// #if CMAA2_UAV_STORE_TYPED
// #if CMAA2_UAV_STORE_TYPED_UNORM_FLOAT
// RWTexture2D<unorm float4>       g_inout_colorWriteonly               : register( u0 );       // final output color
// #else
// RWTexture2D<lpfloat4>           g_inout_colorWriteonly               : register( u0 );       // final output color
// #endif
// #else
// RWTexture2D<uint>               g_inout_colorWriteonly               : register( u0 );       // final output color
// #endif

// #if CMAA2_EDGE_UNORM
// RWTexture2D<unorm float>        g_workingEdges                      : register( u1 );       // output edges (only used in the fist pass)
// #else
// RWTexture2D<uint>               g_workingEdges                      : register( u1 );       // output edges (only used in the fist pass)
// #endif

// RWStructuredBuffer<uint>        g_workingShapeCandidates            : register( u2 );
// RWStructuredBuffer<uint>        g_workingDeferredBlendLocationList  : register( u3 );
// RWStructuredBuffer<uint2>       g_workingDeferredBlendItemList      : register( u4 );       // 
// RWTexture2D<uint>               g_workingDeferredBlendItemListHeads : register( u5 );
// RWByteAddressBuffer             g_workingControlBuffer              : register( u6 );
// RWByteAddressBuffer             g_workingExecuteIndirectBuffer      : register( u7 );

// #if CMAA_MSAA_SAMPLE_COUNT > 1
// Texture2DArray<lpfloat4>        g_inColorMSReadonly                 : register( t2 );       // input MS color
// Texture2D<lpfloat>              g_inColorMSComplexityMaskReadonly   : register( t1 );       // input MS color control surface
// #else
// Texture2D<lpfloat4>             g_inout_colorReadonly                : register( t0 );       // input color
// #endif

// #if CMAA2_EDGE_DETECTION_LUMA_PATH == 2
// Texture2D<float>                g_inLumaReadonly                    : register( t3 );
// #endif
@group(0) @binding(0)
var gather_point_clamp_sampler: sampler;
#ifdef CMAA2_UAV_STORE_TYPED
#ifdef CMAA2_UAV_STORE_TYPED_UNORM_FLOAT
@group(0) @binding(1)
var inout_color_writeonly: texture_storage_2d<f32, write>; // final output colour: TODO ifdef support formats?
#else
@group(0) @binding(1)
var inout_color_writeonly: texture_storage_2d<vec4<f32>, write>; // final output colour: TODO ifdef support formats?
#endif
#else
@group(0) @binding(1)
var inout_color_writeonly: texture_storage_2d<u32, write>; // final output colour: TODO ifdef support formats?
#endif
#ifdef CMAA2_EDGE_UNORM
@group(0) @binding(2)
var working_edges: texture_storage_2d<f32, write>; // output edges (only used in the first pass)
#else
@group(0) @binding(2)
var working_edges: texture_storage_2d<u32, write>; // output edges (only used in the first pass)
#endif
@group(0) @binding(3)
var<storage, read_write> working_shape_candidates: array<u32>;
@group(0) @binding(4)
var<storage, read_write> working_deferred_blend_location_list: array<u32>;
@group(0) @binding(5)
var<storage, read_write> working_deferred_blend_item_list: array<vec2<u32>>; //
@group(0) @binding(6)
var<storage, read_write> working_deferred_blend_item_list_heads: array<atomic<u32>>;    // texture_storage_2d<u32, read_write>;
@group(0) @binding(7)
var<storage, read_write> working_control_buffer: array<atomic<u32>>;
@group(0) @binding(8)
var<storage, read_write> working_execute_indirect_buffer: array<atomic<u32>>;
@group(0) @binding(9)
var inout_color_readonly : texture_2d<f32>; // input colour
// #if CMAA2_EDGE_DETECTION_LUMA_PATH == 2
// @group(0) @binding(10)
// var in_luma_readonly: texture_2d<f32>;
// #endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// encoding/decoding of various data such as edges
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// how .rgba channels from the edge texture maps to pixel edges:
//
//                   A - 0x08               (A - there's an edge between us and a pixel above us)
//              |---------|                 (R - there's an edge between us and a pixel to the right)
//              |         |                 (G - there's an edge between us and a pixel at the bottom)
//     0x04 - B |  pixel  | R - 0x01        (B - there's an edge between us and a pixel to the left)
//              |         |
//              |_________|
//                   G - 0x02
fn PackEdges(edges: vec4<f32>) -> u32  // input edges are binary 0 or 1
{
    return u32(dot(edges, vec4<f32>(1.0, 2.0, 4.0, 8.0)));
}
fn UnpackEdges(value: u32) -> vec4<u32>
{
    var ret: vec4<u32>;
    ret.x = u32((i32(value) & 0x01) != 0);
    ret.y = u32((i32(value) & 0x02) != 0);
    ret.z = u32((i32(value) & 0x04) != 0);
    ret.w = u32((i32(value) & 0x08) != 0);
    return ret;
}
fn UnpackEdgesFlt(value: u32) -> vec4<f32>
{
    var ret: vec4<f32>;
    ret.x = f32((i32(value) & 0x01) != 0);
    ret.y = f32((i32(value) & 0x02) != 0);
    ret.z = f32((i32(value) & 0x04) != 0);
    ret.w = f32((i32(value) & 0x08) != 0);
    return ret;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// source color & color conversion helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


fn LoadSourceColor( pixel_pos: vec2<u32>, offset: vec2<i32>, sample_index: u32) -> vec3<f32>
{
// #if CMAA_MSAA_SAMPLE_COUNT > 1
//     vec3<f32> color = g_inColorMSReadonly.Load( int4( pixel_pos, sampleIndex, 0 ), offset ).rgb;
// #else
    // let color: vec3<f32>  = inout_color_readonly.Load(int3(pixel_pos, 0), offset).rgb;
    let pos = vec2<i32>(pixel_pos) + offset;
    let color: vec3<f32>  = textureLoad(inout_color_readonly, pos, 0).rgb;
// #endif
    return color;
}
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// (R11G11B10 conversion code below taken from Miniengine's PixelPacking_R11G11B10.hlsli,  
// Copyright (c) Microsoft, MIT license, Developed by Minigraph, Author:  James Stanard; original file link:
// https://github.com/Microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli )
//
// The standard 32-bit HDR color format.  Each float has a 5-bit exponent and no sign bit.
fn Pack_R11G11B10_FLOAT(rgb: vec3<f32>) -> u32
{
    // Clamp upper bound so that it doesn't accidentally round up to INF 
    // Exponent=15, Mantissa=1.11111
    let rgb = min(rgb, vec3<f32>(f32(0x477C0000)));  
    let r: u32 = ((f32tof16(rgb.x) + 8) >> 4) & 0x000007FF;
    let g: u32 = ((f32tof16(rgb.y) + 8) << 7) & 0x003FF800;
    let b: u32 = ((f32tof16(rgb.z) + 16) << 17) & 0xFFC00000;
    return r | g | b;
}

fn Unpack_R11G11B10_FLOAT(rgb: u32) -> vec3<f32>
{
    let r: f32 = f16tof32((rgb << 4 ) & 0x7FF0);
    let g: f32 = f16tof32((rgb >> 7 ) & 0x7FF0);
    let b: f32 = f16tof32((rgb >> 17) & 0x7FE0);
    return vec3<f32>(r, g, b);
}

// These next two encodings are great for LDR data.  By knowing that our values are [0.0, 1.0]
// (or [0.0, 2.0), incidentally), we can reduce how many bits we need in the exponent.  We can
// immediately eliminate all postive exponents.  By giving more bits to the mantissa, we can
// improve precision at the expense of range.  The 8E3 format goes one bit further, quadrupling
// mantissa precision but increasing smallest exponent from -14 to -6.  The smallest value of 8E3
// is 2^-14, while the smallest value of 7E4 is 2^-21.  Both are smaller than the smallest 8-bit
// sRGB value, which is close to 2^-12.
//
// This is like R11G11B10_FLOAT except that it moves one bit from each exponent to each mantissa.
fn Pack_R11G11B10_E4_FLOAT(rgb: vec3<f32>) -> u32
{
    // Clamp to [0.0, 2.0).  The magic number is 1.FFFFF x 2^0.  (We can't represent hex floats in HLSL.)
    // This trick works because clamping your exponent to 0 reduces the number of bits needed by 1.
    let rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(f32(0x3FFFFFFF)));
    let r: u32 = ((f32tof16(rgb.r) + 4) >> 3 ) & 0x000007FF;
    let g: u32 = ((f32tof16(rgb.g) + 4) << 8 ) & 0x003FF800;
    let b: u32 = ((f32tof16(rgb.b) + 8) << 18) & 0xFFC00000;
    return r | g | b;
}
//
fn Unpack_R11G11B10_E4_FLOAT(rgb: u32) -> vec3<f32>
{
    let r: f32 = f16tof32((rgb << 3 ) & 0x3FF8);
    let g: f32 = f16tof32((rgb >> 8 ) & 0x3FF8);
    let b: f32 = f16tof32((rgb >> 18) & 0x3FF0);
    return vec3<f32>(r, g, b);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is for temporary storage - R11G11B10_E4 covers 8bit per channel sRGB well enough; 
// For HDR range (CMAA2_SUPPORT_HDR_COLOR_RANGE) use standard float packing - not using it by default because it's not precise 
// enough to match sRGB 8bit, but in a HDR scenario we simply need the range.
// For even more precision un LDR try E3 version and there are other options for HDR range (see above 
// PixelPacking_R11G11GB10.hlsli link for a number of excellent options).
// It's worth noting that since CMAA2 works on high contrast edges, the lack of precision will not be nearly as
// noticeable as it would be on gradients (which always remain unaffected).
fn InternalUnpackColor(packed_color: u32) -> vec3<f32>
{
#ifdef CMAA2_SUPPORT_HDR_COLOR_RANGE
    // ideally using 32bit packing is best for performance reasons but there might be precision issues
    return Unpack_R11G11B10_FLOAT( packed_color );
#else
    return Unpack_R11G11B10_E4_FLOAT( packed_color );
#endif
}

fn InternalPackColor(color: vec3<f32>) -> u32
{
#ifdef CMAA2_SUPPORT_HDR_COLOR_RANGE
    return Pack_R11G11B10_FLOAT(color);
#else
    return Pack_R11G11B10_E4_FLOAT( color );
#endif
}

fn StoreColorSample(pixel_pos: vec2<u32>, color: vec3<f32>, is_complex_shape: bool, msaa_sample_index: u32)
{
    let counter_index: u32 = working_control_buffer.atomicAdd(4*12, 1);

    // quad coordinates
    let quad_pos: vec2<u32> = pixel_pos / vec2<u32>(2u, 2u);
    // 2x2 inter-quad coordinates
    let offset_xy: u32 = (pixel_pos.y % 2u) * 2u + (pixel_pos.x % 2u);
    // encode item-specific info: {2 bits for 2x2 quad location}, {3 bits for MSAA sample index}, {1 bit for is_complex_shape flag}, {26 bits left for address (index)}
    let header: u32 = (offset_xy << 30u) | (msaa_sample_index << 27u) | (u32(is_complex_shape) << 26u);

    let counter_index_with_header: u32 = counter_index | header;
    
    let original_index: u32 = atomicExchange(&working_deferred_blend_item_list_heads[quad_pos], counter_index_with_header);
    working_deferred_blend_item_list[counter_index] = vec2<u32>(original_index, InternalPackColor(color));

    // First one added?
    if(original_index == 0xFFFFFFFFu)
    {
        // Make a list of all edge pixels - these cover all potential pixels where AA is applied.
        let edgelist_counter: u32 = working_control_buffer.atomicAdd(4*8, 1);
        working_deferred_blend_location_list[edgelist_counter] = (quad_pos.x << 16u) | quad_pos.y;
    }
}

#ifdef CMAA2_COLLECT_EXPAND_BLEND_ITEMS
#define CMAA2_BLEND_ITEM_SLM_SIZE 768         // there's a fallback for extreme cases (observed with this value set to 256 or below) in which case image will remain correct but performance will suffer
// groupshared uint        g_groupSharedBlendItemCount;
var<workgroup> group_shared_blend_item_count: atomic<u32>;
// groupshared uint2       g_groupSharedBlendItems[ CMAA2_BLEND_ITEM_SLM_SIZE ];
var<workgroup> group_shared_blend_items: array<vec2<u32>, CMAA2_BLEND_ITEM_SLM_SIZE>;
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Untyped UAV store packing & sRGB conversion helpers
//
fn LINEAR_to_SRGB(val: f32) -> f32
{
    var val = val;
    if( val < 0.0031308 ) {
        val *= 12.92;
    }
    else {
        val = 1.055 * pow(abs(val), 1.0 / 2.4) - 0.055;
    }
    return val;
}

fn LINEAR3_to_SRGB(val: vec3<f32>) -> vec3<f32>
{
    return vec3<f32>( LINEAR_to_SRGB( val.x ), LINEAR_to_SRGB( val.y ), LINEAR_to_SRGB( val.z ) );
}
//
fn FLOAT4_to_R8G8B8A8_UNORM(unpacked_input: vec4<f32>) -> u32
{
    // TODO: can I use a builtin packing function for this?
    return ((u32(saturate(unpacked_input.x) * 255.0 + 0.5)) |
            (u32(saturate(unpacked_input.y) * 255.0 + 0.5) << 8u) |
            (u32(saturate(unpacked_input.z) * 255.0 + 0.5) << 16u) |
            (u32(saturate(unpacked_input.w) * 255.0 + 0.5) << 24u));
}
//
fn FLOAT4_to_R10G10B10A2_UNORM(unpacked_input: vec4<f32>) -> u32
{
    return ((u32(saturate(unpacked_input.x) * 1023.0 + 0.5)) |
            (u32(saturate(unpacked_input.y) * 1023.0 + 0.5) << 10u) |
            (u32(saturate(unpacked_input.z) * 1023.0 + 0.5) << 20u) |
            (u32(saturate(unpacked_input.w) * 3.0 + 0.5) << 30u) );
}
//
// This handles various permutations for various formats with no/partial/full typed UAV store support
fn FinalUAVStore(pixel_pos: vec2<u32>, color: vec3<f32>)
{
#ifdef CMAA2_UAV_STORE_CONVERT_TO_SRGB
    color = LINEAR_to_SRGB( color ) ;
#endif

#ifdef CMAA2_UAV_STORE_TYPED
    textureStore(inout_color_writeonly, vec2<i32>(pixel_pos), vec4<f32>(color, 0.0));
    // inout_color_writeonly[pixel_pos] = vec4<f32>( color.rgb, 0 );
#else
    // #if CMAA2_UAV_STORE_UNTYPED_FORMAT == 1     // R8G8B8A8_UNORM (or R8G8B8A8_UNORM_SRGB with CMAA2_UAV_STORE_CONVERT_TO_SRGB)
    textureStore(inout_color_writeonly, vec2<i32>(pixel_pos), FLOAT4_to_R8G8B8A8_UNORM( vec4<f32>(color, 0.0)));
    #else if CMAA2_UAV_STORE_UNTYPED_FORMAT == 2   // R10G10B10A2_UNORM (or R10G10B10A2_UNORM_SRGB with CMAA2_UAV_STORE_CONVERT_TO_SRGB)
    textureStore(inout_color_writeonly, vec2<i32>(pixel_pos), FLOAT4_to_R10G10B10A2_UNORM( vec4<f32>(color, 0.0)));
    // #else
        // #error CMAA color packing format not defined - add it here!
    #endif
#endif
}
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Edge detection and local contrast adaptation helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
fn GetActualEdgeThreshold() -> f32
{
    var retVal: f32 = g_CMAA2_EdgeThreshold;
#ifdef CMAA2_SCALE_QUALITY_WITH_MSAA
    retVal *= 1.0 + (CMAA_MSAA_SAMPLE_COUNT-1) * 0.06;
#endif
    return retVal;
}
//
fn EdgeDetectColorCalcDiff(color_a: vec3<f32>, color_b: vec3<f32>) -> f32
{
    const LumWeights = vec3<f32>(0.299, 0.587, 0.114);
    let diff = abs((color_a.rgb - color_b.rgb));
    return dot(diff.rgb, LumWeights.rgb);
}
//
// apply custom curve / processing to put input color (linear) in the format required by ComputeEdge
fn ProcessColorForEdgeDetect(color: vec3<f32>) -> vec3<f32>
{
    //pixelColors[i] = LINEAR_to_SRGB( pixelColors[i] );            // correct reference
    //pixelColors[i] = pow( max( 0, pixelColors[i], 1.0 / 2.4 ) );  // approximate sRGB curve
    return sqrt(color); // just very roughly approximate RGB curve
}
//
fn ComputeEdge(x: i32, y: i32, pixel_colors: array<vec3<f32>, 8>) -> vec2<f32>
{
    var temp: vec2<f32>;
    temp.x = EdgeDetectColorCalcDiff(pixel_colors[x + y * 3].rgb, pixel_colors[x + 1 + y * 3].rgb);
    temp.y = EdgeDetectColorCalcDiff(pixel_colors[x + y * 3].rgb, pixel_colors[x + ( y + 1 ) * 3].rgb);
    return temp;    // for HDR edge detection it might be good to premultiply both of these by some factor - otherwise clamping to 1 might prevent some local contrast adaptation. It's a very minor nitpick though, unlikely to significantly affect things.
}                                     
// color -> log luma-for-edges conversion
fn RGBToLumaForEdges(linear_rgb: vec3<f32>) -> f32
{
// #if 0
    // this matches Miniengine luma path
    // let Luma = dot( linear_rgb, vec3<f32>(0.212671, 0.715160, 0.072169));
    // return log2(1.0 + Luma * 15.0) / 4.0;
// #else
    // this is what original FXAA (and consequently CMAA2) use by default - these coefficients correspond to Rec. 601 and those should be
    // used on gamma-compressed components (see https://en.wikipedia.org/wiki/Luma_(video)#Rec._601_luma_versus_Rec._709_luma_coefficients), 
    let luma: f32 = dot(sqrt(linear_rgb.rgb), vec3<f32>(0.299, 0.587, 0.114));  // http://en.wikipedia.org/wiki/CCIR_601
    // using sqrt luma for now but log luma like in miniengine provides a nicer curve on the low-end
    return luma;
// #endif
}
fn ComputeEdgeLuma(x: i32, y: i32, pixel_lumas: array<f32, 8>) -> vec2<f32>
{
    var temp: vec2<f32>;
    temp.x = abs( pixel_lumas[x + y * 3] - pixel_lumas[x + 1 + y * 3] );
    temp.y = abs( pixel_lumas[x + y * 3] - pixel_lumas[x + ( y + 1 ) * 3] );
    return temp;    // for HDR edge detection it might be good to premultiply both of these by some factor - otherwise clamping to 1 might prevent some local contrast adaptation. It's a very minor nitpick though, unlikely to significantly affect things.
}
//
fn ComputeLocalContrastV(x: i32, y: i32, neighbourhood: array<array<vec2<f32>, 4>, 4>) -> f32
{
    // new, small kernel 4-connecting-edges-only local contrast adaptation
    return max(max(neighbourhood[x + 1][y + 0].y, neighbourhood[x + 1][y + 1].y), max(neighbourhood[x + 2][y + 0].y, neighbourhood[x + 2][y + 1].y)) * g_CMAA2_LocalContrastAdaptationAmount;

//    // slightly bigger kernel that enhances edges in-line (not worth the cost)
//  return ( max( max( neighbourhood[x + 1][y + 0].y, neighbourhood[x + 1][y + 1].y ), max( neighbourhood[x + 2][y + 0].y, neighbourhood[x + 2][y + 1].y ) ) 
//        - ( neighbourhood[x + 1][y + 0].x + neighbourhood[x + 1][y + 2].x ) * 0.3 ) * lpfloat( g_CMAA2_LocalContrastAdaptationAmount );
}
//
fn ComputeLocalContrastH(x: i32, y: i32, neighbourhood: array<array<vec2<f32>, 4>, 4>) -> f32
{
    // new, small kernel 4-connecting-edges-only local contrast adaptation
    return max(max(neighbourhood[x + 0][y + 1].x, neighbourhood[x + 1][y + 1].x), max(neighbourhood[x + 0][y + 2].x, neighbourhood[x + 1][y + 2].x)) * g_CMAA2_LocalContrastAdaptationAmount;

//    // slightly bigger kernel that enhances edges in-line (not worth the cost)
//    return ( max( max( neighbourhood[x + 0][y + 1].x, neighbourhood[x + 1][y + 1].x ), max( neighbourhood[x + 0][y + 2].x, neighbourhood[x + 1][y + 2].x ) ) 
//        - ( neighbourhood[x + 0][y + 1].y + neighbourhood[x + 2][y + 1].y ) * 0.3 ) * lpfloat( g_CMAA2_LocalContrastAdaptationAmount );
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

fn ComputeSimpleShapeBlendValues(edges: vec4<f32>, edges_left: vec4<f32>, edges_right: vec4<f32>, edges_top: vec4<f32>, edges_bottom: vec4<f32>, dont_test_shape_validity: bool) -> vec4<f32>
{
    // a 3x3 kernel for higher quality handling of L-based shapes (still rather basic and conservative)
    var from_right = edges.r;
    var from_below = edges.g;
    var from_left = edges.b;
    var from_above = edges.a;

    var blur_coeff: f32 = g_CMAA2_SimpleShapeBlurinessAmount;

    let number_of_edges = u32(dot(edges, vec4<f32>(1.0)));

    let number_of_edges_all_around = dot(edges_left.bga + edges_right.rga + edges_top.rba + edges_bottom.rgb, vec3<f32>(1.0));

    // skip if already tested for before calling this function
    if !dont_test_shape_validity
    {
        // No blur for straight edge
        if number_of_edges == 1u {
            blur_coeff = 0.0;
        }
        // L-like step shape ( only blur if it's a corner, not if it's two parallel edges)
        if number_of_edges == 2u {
            blur_coeff *= ((1.0 - from_below * from_above) * (1.0 - from_right * from_left));
        }
    }

    // L-like step shape
    //[branch]
    if number_of_edges == 2u
    {
        blur_coeff *= 0.75;

// #if 1
        let k = 0.9;
// #if 0
//         from_right   += k * (edges.g * edges_top.r +      edges.a * edges_bottom.r );
//         from_below   += k * (edges.r * edges_left.g +     edges.b * edges_right.g );
//         from_left    += k * (edges.g * edges_top.b +      edges.a * edges_bottom.b );
//         from_above   += k * (edges.b * edges_right.a +    edges.r * edges_left.a );
// #else
        from_right   += k * (edges.g * edges_top.r     * (1.0-edges_left.g)   +     edges.a * edges_bottom.r   * (1.0-edges_left.a));
        from_below   += k * (edges.b * edges_right.g   * (1.0-edges_top.b)    +     edges.r * edges_left.g     * (1.0-edges_top.r));
        from_left    += k * (edges.a * edges_bottom.b  * (1.0-edges_right.a)  +     edges.g * edges_top.b      * (1.0-edges_right.g));
        from_above   += k * (edges.r * edges_left.a    * (1.0-edges_bottom.r) +     edges.b * edges_right.a    * (1.0-edges_bottom.b));
// #endif
// #endif
    }

    // if( number_of_edges == 3 )
    //     blur_coeff *= 0.95;

    // Dampen the blurring effect when lots of neighbouring edges - additionally preserves text and texture detail
#ifdef CMAA2_EXTRA_SHARPNESS
    blur_coeff *= saturate( 1.15 - number_of_edges_all_around / 8.0 );
#else
    blur_coeff *= saturate( 1.30 - number_of_edges_all_around / 10.0 );
#endif

    return vec4<f32>(from_left, from_above, from_right, from_below ) * blur_coeff;
}

fn LoadEdge(pixel_pos: vec2<u32>, offset: vec2<i32>, msaa_sample_index: u32) -> u32
{
// #if CMAA_MSAA_SAMPLE_COUNT > 1
    var edge: u32 = textureLoad(working_edges, vec2<i32>(pixel_pos) + offset).x;
    edge = (edge >> (msaa_sample_index * 4u)) & 0xFu;
// #else
#ifdef CMAA_PACK_SINGLE_SAMPLE_EDGE_TO_HALF_WIDTH
    let a: u32 = u32(pixel_pos.x + offset.x) % 2u;
#ifdef CMAA2_EDGE_UNORM
    var edge: u32 = u32(textureLoad(working_edges, vec2<i32>((pixel_pos.x + offset.x) / 2, pixel_pos.y + offset.y), 0).x * 255.0 + 0.5);
#else    
    // var edge: u32 = g_workingEdges.Load( uint2( uint(pixel_pos.x+offset.x)/2, pixel_pos.y + offset.y ) ).x;
    var edge: u32 = textureLoad(working_edges, vec2<i32>((pixel_pos.x + offset.x) / 2, pixel_pos.y + offset.y), 0).x;
#endif
    edge = (edge >> (a * 4u)) & 0xFu;
#else
    var edge: u32 = textureLoad(working_edges, pixel_pos + offset, 0).x;
#endif
#endif
    return edge;
}

// groupshared lpfloat4 g_groupShared2x2FracEdgesH[CMAA2_CS_INPUT_KERNEL_SIZE_X * CMAA2_CS_INPUT_KERNEL_SIZE_Y];
// groupshared lpfloat4 g_groupShared2x2FracEdgesV[CMAA2_CS_INPUT_KERNEL_SIZE_X * CMAA2_CS_INPUT_KERNEL_SIZE_Y];
var<workgroup> group_shared_2x2_frac_edges_h: array<vec4<f32>, CMAA2_CS_INPUT_KERNEL_SIZE_X * CMAA2_CS_INPUT_KERNEL_SIZE_Y>;
var<workgroup> group_shared_2x2_frac_edges_v: array<vec4<f32>, CMAA2_CS_INPUT_KERNEL_SIZE_X * CMAA2_CS_INPUT_KERNEL_SIZE_Y>;
// void GroupsharedLoadQuadH( uint addr, out lpfloat e00, out lpfloat e10, out lpfloat e01, out lpfloat e11 ) { lpfloat4 val = g_groupShared2x2FracEdgesH[addr]; e00 = val.x; e10 = val.y; e01 = val.z; e11 = val.w; }
// void GroupsharedLoadQuadV( uint addr, out lpfloat e00, out lpfloat e10, out lpfloat e01, out lpfloat e11 ) { lpfloat4 val = g_groupShared2x2FracEdgesV[addr]; e00 = val.x; e10 = val.y; e01 = val.z; e11 = val.w; }
fn GroupsharedLoadQuadHV(addr: u32, e00: ptr<function, vec2<f32>>, e10: ptr<function, vec2<f32>>, e01: ptr<function, vec2<f32>>, e11: ptr<function, vec2<f32>> ) 
{ 
    var valH: vec4<f32> = group_shared_2x2_frac_edges_h[addr]; (*e00).y = valH.x; (*e10).y = valH.y; (*e01).y = valH.z; (*e11).y = valH.w; 
    var valV: vec4<f32> = group_shared_2x2_frac_edges_v[addr]; (*e00).x = valV.x; (*e10).x = valV.y; (*e01).x = valV.z; (*e11).x = valV.w; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Edge detection compute shader
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@compute @workgroup_size(CMAA2_CS_INPUT_KERNEL_SIZE_X, CMAA2_CS_INPUT_KERNEL_SIZE_Y, 1)
fn EdgesColor2x2CS(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_id) group_thread_id: vec3<u32>)
{
    // screen position in the input (expanded) kernel (shifted one 2x2 block up/left)
    // TODO: should maybe be vec2<i32>?
    var pixel_pos: vec2<u32> = group_id.xy * vec2<u32>(CMAA2_CS_OUTPUT_KERNEL_SIZE_X, CMAA2_CS_OUTPUT_KERNEL_SIZE_Y) + group_thread_id.xy - vec2<u32>(1u, 1u);
    pixel_pos *= vec2<u32>(2u, 2u);

    const qe_offsets = array<vec2<u32>, 4>(vec2<u32>(0u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u));
    const row_stride_2x2: u32 = CMAA2_CS_INPUT_KERNEL_SIZE_X;
    const center_addr_2x2: u32 = group_thread_id.x + group_thread_id.y * row_stride_2x2;
    // const uint msaaSliceStride2x2   = CMAA2_CS_INPUT_KERNEL_SIZE_X * CMAA2_CS_INPUT_KERNEL_SIZE_Y;
    const in_output_kernel: bool       = !any( vec4<bool>( group_thread_id.x == (CMAA2_CS_INPUT_KERNEL_SIZE_X - 1u), group_thread_id.x == 0u, group_thread_id.y == (CMAA2_CS_INPUT_KERNEL_SIZE_Y - 1u), group_thread_id.y == 0u) );

    var i: u32;
    var qe0: vec2<f32>;
    var qe1: vec2<f32>;
    var qe2: vec2<f32>;
    var qe3: vec2<f32>;
    var out_edges = vec4<u32>(0u);

// TODO: MSAA support
// #if CMAA_MSAA_SAMPLE_COUNT > 1
//     var firstLoopIsEnough = false;

//     // #ifdef CMAA_MSAA_USE_COMPLEXITY_MASK
//     {
//         var texSize: vec2<f32>;
//         g_inColorMSComplexityMaskReadonly.GetDimensions( texSize.x, texSize.y );
//         let gather_uv = vec2<f32>(pixel_pos) / texSize;
//         let TL: vec4<f32> = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gather_uv, int2( 0, 0 ) );
//         let TR: vec4<f32> = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gather_uv, int2( 2, 0 ) );
//         let BL: vec4<f32> = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gather_uv, int2( 0, 2 ) );
//         let BR: vec4<f32> = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gather_uv, int2( 2, 2 ) );
//         let sumAll: vec4<f32> = TL+TR+BL+BR;
//         firstLoopIsEnough = !any(sumAll);
//     }
//     // #endif
// #endif


//     // not optimal - to be optimized
// #if CMAA_MSAA_SAMPLE_COUNT > 1
//     // clear this here to reduce complexity below - turns out it's quicker as well this way
//     g_workingDeferredBlendItemListHeads[ uint2( pixel_pos ) / 2 ] = 0xFFFFFFFF;
//     [loop]
//     for( uint msaa_sample_idx = 0; msaa_sample_idx < CMAA_MSAA_SAMPLE_COUNT; msaa_sample_idx++ )
//     {
//         bool msaaSampleIsRelevant = !firstLoopIsEnough || msaa_sample_idx == 0;
//         [branch]
//         if( msaaSampleIsRelevant )
//         {
// #else
//         {
//             uint msaa_sample_idx = 0;
// #endif

    let msaa_sample_idx = 0u;

            // edge detection
// #if CMAA2_EDGE_DETECTION_LUMA_PATH == 0
            var pixel_colors: array<vec3<f32>, 8>;
            for(var i = 0; i < 8; i++) {

                pixel_colors[i] = LoadSourceColor(pixel_pos, vec2<i32>(i % 3, i / 3), msaa_sample_idx).rgb;
            }

            for(var i = 0; i < 8; i++) {
                pixel_colors[i] = ProcessColorForEdgeDetect(pixel_colors[i]);
            }

            qe0 = ComputeEdge(0, 0, pixel_colors);
            qe1 = ComputeEdge(1, 0, pixel_colors);
            qe2 = ComputeEdge(0, 1, pixel_colors);
            qe3 = ComputeEdge(1, 1, pixel_colors);
// #else // CMAA2_EDGE_DETECTION_LUMA_PATH != 0
            var pixel_lumas: array<f32, 8>;
    // #if CMAA2_EDGE_DETECTION_LUMA_PATH == 1 // compute in-place
            for(var i = 0; i < 8; i++)
            {
                let color = LoadSourceColor(pixel_pos, vec2<i32>(i % 3, i / 3), msaa_sample_idx).rgb;
                pixel_lumas[i] = RGBToLumaForEdges( color );
            }
    // // #else if CMAA2_EDGE_DETECTION_LUMA_PATH == 2 // source from outside
    // #if 0 // same as below, just without Gather
    //         for( i = 0; i < 3 * 3 - 1; i++ )
    //              pixel_lumas[i] = g_inLumaReadonly.Load( int3( pixel_pos, 0 ), int2( i % 3, i / 3 ) ).r;
    // #else
            let tex_size: vec2<f32> = vec2<f32>(textureDimensions(g_inLumaReadonly));
            let gather_uv = (vec2<f32>(pixel_pos) + vec2<f32>(0.5, 0.5)) / tex_size;
            let TL: vec4<f32> = textureGather(0, g_inLumaReadonly, gather_point_clamp_sampler, gather_uv);
            let TR: vec4<f32> = textureGather(0, g_inLumaReadonly, gather_point_clamp_sampler, gather_uv, vec2<i32>(1, 0));
            let BL: vec4<f32> = textureGather(0, g_inLumaReadonly, gather_point_clamp_sampler, gather_uv, vec2<i32>(0, 1));
            pixel_lumas[0] = TL.w; pixel_lumas[1] = TL.z; pixel_lumas[2] = TR.z; pixel_lumas[3] = TL.x;
            pixel_lumas[4] = TL.y; pixel_lumas[5] = TR.y; pixel_lumas[6] = BL.x; pixel_lumas[7] = BL.y;
    #endif
    // #else if CMAA2_EDGE_DETECTION_LUMA_PATH == 3 // source in alpha channel of input color
            let tex_size: vec2<f32> = vec2<f32>(textureDimensions(inout_color_readonly));
            let gather_uv = (vec2<f32>(pixel_pos) + vec2<f32>(0.5, 0.5)) / tex_size;
            let TL: vec4<f32> = textureGather(3, inout_color_readonly, gather_point_clamp_sampler, gather_uv);
            let TR: vec4<f32> = textureGather(3, inout_color_readonly, gather_point_clamp_sampler, gather_uv, vec2<i32>(1, 0) );
            let BL: vec4<f32> = textureGather(3, inout_color_readonly, gather_point_clamp_sampler, gather_uv, vec2<i32>(0, 1) );
            pixel_lumas[0] = TL.w; pixel_lumas[1] = TL.z; pixel_lumas[2] = TR.z; pixel_lumas[3] = TL.x; 
            pixel_lumas[4] = TL.y; pixel_lumas[5] = TR.y; pixel_lumas[6] = BL.x; pixel_lumas[7] = BL.y;                 
    #endif
            qe0 = ComputeEdgeLuma( 0, 0, pixel_lumas );
            qe1 = ComputeEdgeLuma( 1, 0, pixel_lumas );
            qe2 = ComputeEdgeLuma( 0, 1, pixel_lumas );
            qe3 = ComputeEdgeLuma( 1, 1, pixel_lumas );
#endif
            // g_groupShared2x2FracEdgesV[centerAddr2x2 + rowStride2x2 * 0] = lpfloat4( qe0.x, qe1.x, qe2.x, qe3.x );
            // g_groupShared2x2FracEdgesH[centerAddr2x2 + rowStride2x2 * 0] = lpfloat4( qe0.y, qe1.y, qe2.y, qe3.y );
            group_shared_2x2_frac_edges_v[center_addr_2x2] = vec4<f32>( qe0.x, qe1.x, qe2.x, qe3.x );
            group_shared_2x2_frac_edges_h[center_addr_2x2] = vec4<f32>( qe0.y, qe1.y, qe2.y, qe3.y );
     
// #if CMAA_MSAA_SAMPLE_COUNT > 1
//          }  // if (msaaSampleIsRelevant)
// #endif

        workgroupBarrier();

        if in_output_kernel {
            var top_row: vec2<f32> = group_shared_2x2_frac_edges_v[center_addr_2x2 - row_stride_2x2].zw; // top row's bottom edge
            var left_column: vec2<f32> = group_shared_2x2_frac_edges_h[center_addr_2x2 - 1u].yw;         // left column's right edge

            let some_non_zero_edges: bool = any(vec4<bool>(vec4<f32>(qe0, qe1) + vec4<f32>(qe2, qe3) + vec4<f32>(top_row[0], top_row[1], left_column[0], left_column[1])));
            //bool some_non_zero_edges = packedCenterEdges.x | packedCenterEdges.y | (packedQuadP0M1.y & 0xFFFF0000) | (packedQuadM1P0.x & 0xFF00FF00);

            if some_non_zero_edges {
    // #if CMAA_MSAA_SAMPLE_COUNT == 1
                // Clear deferred color list heads to empty (if potentially needed - even though some edges might get culled by local contrast adaptation 
                // step below, it's still cheaper to just clear it without additional logic)
                // TODO: working_deferred_blend_item_list_heads is supposed to be a texture, but we can't do atomic ops on those in wgpu so this is wrong
                // working_deferred_blend_item_list_heads[vec2<u32>(pixel_pos) / 2u] = 0xFFFFFFFFu;
                atomicStore(&working_deferred_blend_item_list_heads[vec2<u32>(pixel_pos) / 2u], 0xFFFFFFFFu);
    // #endif

                var ce: array<vec4<f32>, 4>;

            // #if 1 // local contrast adaptation
                var dummyd0: vec2<f32>;
                var dummyd1: vec2<f32>;
                var dummyd2: vec2<f32>; //vec2<f32>
                var dummyd3: vec2<f32>; //vec2<f32>
                // var neighbourhood: array<array<ptr<function, vec2<f32>>, 4>, 4>;
                var neighbourhood: array<array<vec2<f32>, 4>, 4>;

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // load & unpack kernel data from SLM
                GroupsharedLoadQuadHV(center_addr_2x2 - row_stride_2x2 - 1u , &dummyd0, &dummyd1, &dummyd2, &neighbourhood[0][0]);
                GroupsharedLoadQuadHV(center_addr_2x2 - row_stride_2x2     , &dummyd0, &dummyd1, &neighbourhood[1][0], &neighbourhood[2][0]);
                GroupsharedLoadQuadHV(center_addr_2x2 - row_stride_2x2 + 1u , &dummyd0, &dummyd1, &neighbourhood[3][0], &dummyd2);
                GroupsharedLoadQuadHV(center_addr_2x2 - 1u                , &dummyd0, &neighbourhood[0][1], &dummyd1, &neighbourhood[0][2]);
                GroupsharedLoadQuadHV(center_addr_2x2 + 1u                , &neighbourhood[3][1], &dummyd0, &neighbourhood[3][2], &dummyd1);
                GroupsharedLoadQuadHV(center_addr_2x2 - 1u + row_stride_2x2 , &dummyd0, &neighbourhood[0][3], &dummyd1, &dummyd2);
                GroupsharedLoadQuadHV(center_addr_2x2 + row_stride_2x2     , &neighbourhood[1][3], &neighbourhood[2][3], &dummyd0, &dummyd1);
                neighbourhood[1][0].y = top_row[0]; // already in registers
                neighbourhood[2][0].y = top_row[1]; // already in registers
                neighbourhood[0][1].x = left_column[0]; // already in registers
                neighbourhood[0][2].x = left_column[1]; // already in registers
                neighbourhood[1][1] = qe0; // already in registers
                neighbourhood[2][1] = qe1; // already in registers
                neighbourhood[1][2] = qe2; // already in registers
                neighbourhood[2][2] = qe3; // already in registers
                // if neighbourhood declaration needs to be a pointer, then the following needs to be uncommented
                // GroupsharedLoadQuadHV(center_addr_2x2 - row_stride_2x2 - 1u , &dummyd0, &dummyd1, &dummyd2, neighbourhood[0][0]);
                // GroupsharedLoadQuadHV(center_addr_2x2 - row_stride_2x2     , &dummyd0, &dummyd1, neighbourhood[1][0], neighbourhood[2][0]);
                // GroupsharedLoadQuadHV(center_addr_2x2 - row_stride_2x2 + 1u , &dummyd0, &dummyd1, neighbourhood[3][0], &dummyd2);
                // GroupsharedLoadQuadHV(center_addr_2x2 - 1u                , &dummyd0, neighbourhood[0][1], &dummyd1, neighbourhood[0][2]);
                // GroupsharedLoadQuadHV(center_addr_2x2 + 1u                , neighbourhood[3][1], &dummyd0, neighbourhood[3][2], &dummyd1);
                // GroupsharedLoadQuadHV(center_addr_2x2 - 1u + row_stride_2x2 , &dummyd0, neighbourhood[0][3], &dummyd1, &dummyd2);
                // GroupsharedLoadQuadHV(center_addr_2x2 + row_stride_2x2     , neighbourhood[1][3], neighbourhood[2][3], &dummyd0, &dummyd1);
                // (*neighbourhood[1][0]).y = top_row[0]; // already in registers
                // (*neighbourhood[2][0]).y = top_row[1]; // already in registers
                // (*neighbourhood[0][1]).x = left_column[0]; // already in registers
                // (*neighbourhood[0][2]).x = left_column[1]; // already in registers
                // (*neighbourhood[1][1]) = qe0; // already in registers
                // (*neighbourhood[2][1]) = qe1; // already in registers
                // (*neighbourhood[1][2]) = qe2; // already in registers
                // (*neighbourhood[2][2]) = qe3; // already in registers
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
                top_row[0]     = f32((top_row[0]     - ComputeLocalContrastH( 0, -1, neighbourhood)) > GetActualEdgeThreshold());
                top_row[1]     = f32((top_row[1]     - ComputeLocalContrastH( 1, -1, neighbourhood)) > GetActualEdgeThreshold());
                left_column[0] = f32((left_column[0] - ComputeLocalContrastV( -1, 0, neighbourhood)) > GetActualEdgeThreshold());
                left_column[1] = f32((left_column[1] - ComputeLocalContrastV( -1, 1, neighbourhood)) > GetActualEdgeThreshold());

                ce[0].x = f32((qe0.x - ComputeLocalContrastV(0, 0, neighbourhood)) > GetActualEdgeThreshold());
                ce[0].y = f32((qe0.y - ComputeLocalContrastH(0, 0, neighbourhood)) > GetActualEdgeThreshold());
                ce[1].x = f32((qe1.x - ComputeLocalContrastV(1, 0, neighbourhood)) > GetActualEdgeThreshold());
                ce[1].y = f32((qe1.y - ComputeLocalContrastH(1, 0, neighbourhood)) > GetActualEdgeThreshold());
                ce[2].x = f32((qe2.x - ComputeLocalContrastV(0, 1, neighbourhood)) > GetActualEdgeThreshold());
                ce[2].y = f32((qe2.y - ComputeLocalContrastH(0, 1, neighbourhood)) > GetActualEdgeThreshold());
                ce[3].x = f32((qe3.x - ComputeLocalContrastV(1, 1, neighbourhood)) > GetActualEdgeThreshold());
                ce[3].y = f32((qe3.y - ComputeLocalContrastH(1, 1, neighbourhood)) > GetActualEdgeThreshold());
            // #else
                top_row[0]     = f32(top_row[0]    > GetActualEdgeThreshold());
                top_row[1]     = f32(top_row[1]    > GetActualEdgeThreshold());
                left_column[0] = f32(left_column[0]> GetActualEdgeThreshold());
                left_column[1] = f32(left_column[1]> GetActualEdgeThreshold());
                ce[0].x = f32(qe0.x > GetActualEdgeThreshold());
                ce[0].y = f32(qe0.y > GetActualEdgeThreshold());
                ce[1].x = f32(qe1.x > GetActualEdgeThreshold());
                ce[1].y = f32(qe1.y > GetActualEdgeThreshold());
                ce[2].x = f32(qe2.x > GetActualEdgeThreshold());
                ce[2].y = f32(qe2.y > GetActualEdgeThreshold());
                ce[3].x = f32(qe3.x > GetActualEdgeThreshold());
                ce[3].y = f32(qe3.y > GetActualEdgeThreshold());
            #endif

                //left
                ce[0].z = left_column[0];
                ce[1].z = ce[0].x;
                ce[2].z = left_column[1];
                ce[3].z = ce[2].x;

                // top
                ce[0].w = top_row[0];
                ce[1].w = top_row[1];
                ce[2].w = ce[0].y;
                ce[3].w = ce[1].y;

                for (var i = 0; i < 4; i++) {
                    const local_pixel_pos: vec2<u32> = pixel_pos + qe_offsets[i];

                    const edges: vec4<f32> = ce[i];

                    // if there's at least one two edge corner, this is a candidate for simple or complex shape processing...
                    let is_candidate: bool = i32(edges.x * edges.y + edges.y * edges.z + edges.z * edges.w + edges.w * edges.x) != 0;
                    if is_candidate {
                        let counter_index: u32 = working_control_buffer.atomicAdd(4 * 4, 1);
                        working_shape_candidates[counter_index] = (local_pixel_pos.x << 18u) | (msaa_sample_idx << 14u) | local_pixel_pos.y;
                    }

                    // Write out edges - we write out all, including empty pixels, to make sure shape detection edge tracing
                    // doesn't continue on previous frame's edges that no longer exist.
                    let packed_edge = PackEdges(edges);
    // #if CMAA_MSAA_SAMPLE_COUNT > 1
                    out_edges[i] |= packed_edge << (msaa_sample_idx * 4u);
    // #else
                    out_edges[i] = packed_edge;
    #endif
                }
            }
        }
    // }

    // finally, write the edges!
    if in_output_kernel {
// #ifdef CMAA_PACK_SINGLE_SAMPLE_EDGE_TO_HALF_WIDTH && if CMAA_MSAA_SAMPLE_COUNT == 1
// #ifdef CMAA2_EDGE_UNORM
        // g_workingEdges[ int2(pixel_pos.x/2, pixel_pos.y+0) ] = ((out_edges[1] << 4) | out_edges[0]) / 255.0;
        // g_workingEdges[ int2(pixel_pos.x/2, pixel_pos.y+1) ] = ((out_edges[3] << 4) | out_edges[2]) / 255.0;
        textureStore(working_edges, vec2<i32>(pixel_pos.x/2, pixel_pos.y+0), (out_edges[1] << 4) | out_edges[0] / 255.0);
        textureStore(working_edges, vec2<i32>(pixel_pos.x/2, pixel_pos.y+1), (out_edges[3] << 4) | out_edges[2] / 255.0);
// #else
        // g_workingEdges[ int2(pixel_pos.x/2, pixel_pos.y+0) ] = (out_edges[1] << 4) | out_edges[0];
        // g_workingEdges[ int2(pixel_pos.x/2, pixel_pos.y+1) ] = (out_edges[3] << 4) | out_edges[2];
        textureStore(working_edges, vec2<i32>(pixel_pos.x/2, pixel_pos.y+0), (out_edges[1] << 4) | out_edges[0]);
        textureStore(working_edges, vec2<i32>(pixel_pos.x/2, pixel_pos.y+1), (out_edges[3] << 4) | out_edges[2]);
#endif
// #else
        {
            for(var i: u32 = 0u; i < 4u; i++) {
                textureStore(working_edges, pixel_pos + qe_offsets[i], out_edges[i]);
                // g_workingEdges[pixel_pos + qe_offsets[i]] = out_edges[i];
            }
        }
#endif
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute shaders used to generate DispatchIndirect() control buffer
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Compute dispatch arguments for the DispatchIndirect() that calls ProcessCandidatesCS and DeferredColorApply2x2CS
@compute @workgroup_size(1, 1, 1)
fn ComputeDispatchArgsCS(@builtin(workgroup_id) group_id: vec3<u32>)
{
    // activated once on Dispatch( 2, 1, 1 )
    if group_id.x == 1u {
        // get current count
        // var shape_candidate_count: u32 = g_workingControlBuffer.Load(4*4);
        var shape_candidate_count: u32 = atomicLoad(&working_control_buffer[16]);


        // check for overflow!
        let append_buffer_max_count: u32 = arrayLength(&working_shape_candidates);
        // g_workingShapeCandidates.GetDimensions( append_buffer_max_count, append_buffer_stride );
        shape_candidate_count = min( shape_candidate_count, append_buffer_max_count );

        // write dispatch indirect arguments for ProcessCandidatesCS
        atomicStore(&working_execute_indirect_buffer[0], (shape_candidate_count + CMAA2_PROCESS_CANDIDATES_NUM_THREADS - 1u) / CMAA2_PROCESS_CANDIDATES_NUM_THREADS);
        atomicStore(&working_execute_indirect_buffer[4], 1u);
        atomicStore(&working_execute_indirect_buffer[8], 1u);
        // g_workingExecuteIndirectBuffer.Store( 4*0, ( shape_candidate_count + CMAA2_PROCESS_CANDIDATES_NUM_THREADS - 1 ) / CMAA2_PROCESS_CANDIDATES_NUM_THREADS );
        // g_workingExecuteIndirectBuffer.Store( 4*1, 1 );                                                                                                       
        // g_workingExecuteIndirectBuffer.Store( 4*2, 1 );                                                                                                       

        // write actual number of items to process in ProcessCandidatesCS
        atomicStore(&working_control_buffer[12], shape_candidate_count);
        // g_workingControlBuffer.Store( 4*3, shape_candidate_count );
    } 
    // activated once on Dispatch( 1, 2, 1 )
    else if group_id.y == 1u {
        // get current count
        // var blend_location_count: u32 = g_workingControlBuffer.Load(4*8);
        var blend_location_count: u32 = atomicLoad(&working_control_buffer[32]);


        // check for overflow!
        { 
            let append_buffer_max_count: u32 = arrayLength(&working_deferred_blend_location_list);
            // g_workingDeferredBlendLocationList.GetDimensions( append_buffer_max_count, append_buffer_stride );
            blend_location_count = min( blend_location_count, append_buffer_max_count );
        }

        // write dispatch indirect arguments for DeferredColorApply2x2CS
#ifdef CMAA2_DEFERRED_APPLY_THREADGROUP_SWAP
        atomicStore(&working_execute_indirect_buffer[0], 1u);
        atomicStore(&working_execute_indirect_buffer[4], (blend_location_count + CMAA2_DEFERRED_APPLY_NUM_THREADS - 1u) / CMAA2_DEFERRED_APPLY_NUM_THREADS);
        // g_workingExecuteIndirectBuffer.Store( 4*0, 1 );
        // g_workingExecuteIndirectBuffer.Store( 4*1, ( blend_location_count + CMAA2_DEFERRED_APPLY_NUM_THREADS - 1 ) / CMAA2_DEFERRED_APPLY_NUM_THREADS );
#else
        atomicStore(&working_execute_indirect_buffer[0], (blend_location_count + CMAA2_DEFERRED_APPLY_NUM_THREADS - 1u) / CMAA2_DEFERRED_APPLY_NUM_THREADS);
        atomicStore(&working_execute_indirect_buffer[4], 1u);
        // g_workingExecuteIndirectBuffer.Store( 4*0, ( blend_location_count + CMAA2_DEFERRED_APPLY_NUM_THREADS - 1 ) / CMAA2_DEFERRED_APPLY_NUM_THREADS );
        // g_workingExecuteIndirectBuffer.Store( 4*1, 1 );
#endif
        // g_workingExecuteIndirectBuffer.Store( 4*2, 1 );
        atomicStore(&working_execute_indirect_buffer[8], 1u);

        // write actual number of items to process in DeferredColorApply2x2CS
        // g_workingControlBuffer.Store( 4*3, blend_location_count);
        working_control_buffer.Store( 4*3, blend_location_count);

        // clear counters for next frame
        atomicStore(&working_control_buffer[16], 0u);
        atomicStore(&working_control_buffer[32], 0u);
        atomicStore(&working_control_buffer[48], 0u);
        // g_workingControlBuffer.Store( 4*4 , 0 );
        // g_workingControlBuffer.Store( 4*8 , 0 );
        // g_workingControlBuffer.Store( 4*12, 0 );
    }
}
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// fn FindZLineLengths( out lpfloat line_length_left, out lpfloat line_length_right, uint2 screen_pos, uniform bool horizontal, uniform bool inverted_z_shape, const float2 step_right, uint msaa_sample_idx )
fn FindZLineLengths(line_length_left: ptr<function, f32>, line_length_right: ptr<function, f32>, screen_pos: vec2<u32>, horizontal: bool, inverted_z_shape: bool, step_right: vec2<f32>, msaa_sample_idx: u32)
{
// this enables additional conservativeness test but is pretty detrimental to the final effect so left disabled by default even when CMAA2_EXTRA_SHARPNESS is enabled
// #define CMAA2_EXTRA_CONSERVATIVENESS2
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TODO: a cleaner and faster way to get to these - a precalculated array indexing maybe?
    var mask_left: u32;
    var bits_continue_left: u32;
    var mask_right: u32;
    var bits_continue_right: u32;
    {
        // Horizontal (vertical is the same, just rotated 90- counter-clockwise)
        // Inverted Z case:              // Normal Z case:
        //   __                          // __
        //  X|                           //  X|
        // --                            //   --
        var mask_trace_left: u32;
        var mask_trace_right: u32;
// #if CMAA2_EXTRA_CONSERVATIVENESS2
        // uint maskStopLeft, maskStopRight;
// #endif
        if(horizontal)
        {
            mask_trace_left = 0x08u; // tracing top edge
            mask_trace_right = 0x02u; // tracing bottom edge
// #if CMAA2_EXTRA_CONSERVATIVENESS2
//             maskStopLeft = 0x01; // stop on right edge
//             maskStopRight = 0x04; // stop on left edge
// #endif
        }
        else
        {
            mask_trace_left = 0x04u; // tracing left edge
            mask_trace_right = 0x01u; // tracing right edge
// #if CMAA2_EXTRA_CONSERVATIVENESS2
//             maskStopLeft = 0x08; // stop on top edge
//             maskStopRight = 0x02; // stop on bottom edge
// #endif
        }
        if(inverted_z_shape)
        {
            let temp = mask_trace_left;
            mask_trace_left = mask_trace_right;
            mask_trace_right = temp;
        }
        mask_left = mask_trace_left;
        bits_continue_left = mask_trace_left;
        mask_right = mask_trace_right;
// #if CMAA2_EXTRA_CONSERVATIVENESS2
//         mask_left |= maskStopLeft;
//         mask_right |= maskStopRight;
// #endif
        bits_continue_right = mask_trace_right;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    var continue_left = true;
    var continue_right = true;
    *line_length_left = 1.0;
    *line_length_right = 1.0;
    for( ; ; )
    {
        let edge_left: u32 = LoadEdge((screen_pos.xy) - vec2<u32>(step_right * f32(*line_length_left)) , vec2<i32>( 0, 0 ), msaa_sample_idx );
        let edge_right: u32 = LoadEdge((screen_pos.xy) + vec2<u32>(step_right * (f32(*line_length_right) + 1.0)), vec2<i32>( 0, 0 ), msaa_sample_idx );

        // stop on encountering 'stopping' edge (as defined by masks)
        continue_left    = continue_left  && ( ( edge_left & mask_left ) == bits_continue_left );
        continue_right   = continue_right && ( ( edge_right & mask_right ) == bits_continue_right );

        *line_length_left += f32(continue_left);
        *line_length_right += f32(continue_right);

        var max_LR: f32 = max(*line_length_right, *line_length_left);

        // both stopped? cause the search end by setting max_LR to max length.
        if(!continue_left && !continue_right) {
            max_LR = f32(c_maxLineLength);
        }

        // either the longer one is ahead of the smaller (already stopped) one by more than a factor of x, or both
        // are stopped - end the search.
#ifdef CMAA2_EXTRA_SHARPNESS
        if( max_LR >= min(f32(c_maxLineLength), (1.20 * min(*line_length_right, *line_length_left) - 0.20))) {
#else
        if( max_LR >= min(f32(c_maxLineLength), (1.25 * min(*line_length_right, *line_length_left) - 0.25))) {
#endif
            break;
        }
    }
}

// these are blendZ settings, determined empirically :)
const c_symmetryCorrectionOffset: f32 = 0.22;
#ifdef CMAA2_EXTRA_SHARPNESS
const c_dampeningEffect: f32 = 0.11;
#else
const c_dampeningEffect: f32 = 0.15;
#endif

#ifdef CMAA2_COLLECT_EXPAND_BLEND_ITEMS
fn CollectBlendZs(screen_pos: vec2<u32>, horizontal: bool, inverted_z_shape: bool, shape_quality_score: f32, line_length_left: f32, line_length_right: f32, step_right: vec2<f32>, msaa_sample_idx: u32) -> bool
{
    let left_odd: f32 = c_symmetryCorrectionOffset * f32(line_length_left % 2.0);
    let right_odd: f32 = c_symmetryCorrectionOffset * f32(line_length_right % 2.0);

    let dampen_effect: f32 = saturate(f32(line_length_left + line_length_right - shape_quality_score) * c_dampeningEffect);

    let loop_from: f32 = -floor((line_length_left + 1.0) / 2.0) + 1.0;
    let loop_to: f32 = floor((line_length_right + 1.0) / 2.0);
    
    const blend_item_count: u32 = u32(loop_to - loop_from + 1.0);
    // InterlockedAdd( g_groupSharedBlendItemCount, blend_item_count, item_idx );
    var item_idx = atomicAdd(&group_shared_blend_item_count, blend_item_count);
    // safety
    if((item_idx + blend_item_count) > CMAA2_BLEND_ITEM_SLM_SIZE) {
        return false;
    }

    let total_length: f32 = (loop_to - loop_from) + 1.0 - left_odd - right_odd;
    let lerp_step: f32 = 1.0 / total_length;

    let lerp_from_k: f32 = (0.5 - left_odd - loop_from) * lerp_step;

    let item_header: u32 = (screen_pos.x << 18u) | (msaa_sample_idx << 14u) | screen_pos.y;
    let item_val_static: u32 = (u32(horizontal) << 31u) | (u32(inverted_z_shape) << 30u);

    for(var i = loop_from; i <= loop_to; i += 1.0) {
        // unused?
        // lpfloat lerpVal = lerp_step * i + lerp_from_k;

        let second_part = f32(i > 0.0);
        let src_offset = 1.0 - second_part * 2.0;

        let lerp_k = ((lerp_step * i + lerp_from_k) * src_offset + second_part) * dampen_effect;

        var encoded_item: vec2<u32>;
        encoded_item.x = item_header;
        encoded_item.y = item_val_static | ((u32(i + 256.0)) << 20u) | ((u32(src_offset + 256.0)) << 10u) | u32(saturate(lerp_k) * 1023.0 + 0.5); //*& 0x3FF*/
        // g_groupSharedBlendItems[item_idx++] = encoded_item;
        // TODO: maybe needs to be after using it
        item_idx += 1u;
        group_shared_blend_items[item_idx] = encoded_item;
    }
    return true;
}
#endif

fn BlendZs(screen_pos: vec2<u32>, horizontal: bool, inverted_z_shape: bool, shape_quality_score: f32, line_length_left: f32, line_length_right: f32, step_right: vec2<f32>, msaaSampleIndex: u32) {
    var blend_dir: vec2<f32>;
    if horizontal {
        blend_dir = vec2<f32>(0.0, -1.0);
    } else {
        blend_dir = vec2<f32>(-1.0, 0.0);
    }

    if inverted_z_shape {
        blend_dir = -blend_dir;
    }

    let left_odd: f32 = c_symmetryCorrectionOffset * line_length_left % 2.0;
    let right_odd: f32 = c_symmetryCorrectionOffset * line_length_right % 2.0;

    let dampen_effect: f32 = saturate((line_length_left + line_length_right - shape_quality_score) * c_dampeningEffect);

    let loop_from = -floor((line_length_left + 1.0) / 2.0) + 1.0;
    let loop_to = floor((line_length_right + 1.0) / 2.0);
    
    let total_length = (loop_to - loop_from) + 1.0 - left_odd - right_odd;
    let lerp_step = 1.0 / total_length;

    let lerp_from_k = (0.5 - left_odd - loop_from) * lerp_step;

    for(var i = loop_from; i <= loop_to; i += 1.0)
    {
        // unused?
        // lpfloat lerpVal = lerp_step * i + lerp_from_k;

        let second_part = f32(i > 0.0);
        let src_offset = 1.0 - second_part * 2.0;

        let lerp_k = ((lerp_step * i + lerp_from_k) * src_offset + second_part) * dampen_effect;

        let pixel_pos = screen_pos + vec2<u32>(step_right * i);

        let color_center = LoadSourceColor(pixel_pos, vec2<i32>(0, 0), msaaSampleIndex).rgb;
        
        let color_from = LoadSourceColor(pixel_pos.xy + vec2<u32>(blend_dir * src_offset), vec2<i32>(0, 0), msaaSampleIndex).rgb;
        
        let output = mix(color_center.rgb, color_from.rgb, lerp_k);

        StoreColorSample(pixel_pos.xy, output, true, msaaSampleIndex);
    }
}

// TODO:
// There were issues with moving this (including the calling code) to half-float on some hardware (broke in certain cases on RX 480).
// Further investigation is required.
// fn DetectZsHorizontal( in lpfloat4 edges, in lpfloat4 edgesM1P0, in lpfloat4 edgesP1P0, in lpfloat4 edgesP2P0, inverted_z_score: ptr<function, f32>, normal_z_score: ptr<function, f32>)
fn DetectZsHorizontal(edges: vec4<f32>, edgesM1P0: vec4<f32>, edgesP1P0: vec4<f32>, edgesP2P0: vec4<f32>, inverted_z_score: ptr<function, f32>, normal_z_score: ptr<function, f32>) {
    // Inverted Z case:
    //   __
    //  X|
    // --
    {
        *inverted_z_score = edges.r * edges.g *edgesP1P0.a;
        *inverted_z_score *= 2.0 + ((edgesM1P0.g + edgesP2P0.a) ) - (edges.a + edgesP1P0.g) - 0.7 * (edgesP2P0.g + edgesM1P0.a + edges.b + edgesP1P0.r);
    }

    // Normal Z case:
    // __
    //  X|
    //   --
    {
        *normal_z_score = edges.r * edges.a * edgesP1P0.g;
        *normal_z_score *= 2.0 + ((edgesM1P0.a + edgesP2P0.g) ) - (edges.g + edgesP1P0.a) - 0.7 * (edgesP2P0.a + edgesM1P0.g + edges.b + edgesP1P0.r);
    }
}

// [numthreads( CMAA2_PROCESS_CANDIDATES_NUM_THREADS, 1, 1 )]
@compute @workgroup_size( CMAA2_PROCESS_CANDIDATES_NUM_THREADS, 1, 1 )
// fn ProcessCandidatesCS( uint3 dispatch_thread_id : SV_dispatch_thread_id, uint3 group_thread_id : SV_group_thread_id )
fn ProcessCandidatesCS(@builtin(global_invocation_id) dispatch_thread_id : vec3<u32>, @builtin(local_invocation_id) group_thread_id: vec3<u32>)
{
#ifdef CMAA2_COLLECT_EXPAND_BLEND_ITEMS
    if group_thread_id.x == 0u {
        atomicStore(group_shared_blend_item_count, 0u);
    }
    workgroupBarrier();
#endif

    var msaaSampleIndex = 0u;
    // const uint num_candidates = g_workingControlBuffer.Load(4*3); //g_workingControlBuffer[3];
    const num_candidates = atomicLoad(&working_control_buffer[12]); //g_workingControlBuffer[3];
    if(dispatch_thread_id.x < num_candidates)
    {
	    // uint pixel_id = g_workingShapeCandidates[dispatch_thread_id.x];
	    let pixel_id = working_shape_candidates[dispatch_thread_id.x];

// #ifdef 0 // debug display
//         let screen_size = textureDimensions(inout_color_readonly);
//         StoreColorSample(vec2<u32>(dispatch_thread_id.x % screen_size.x, dispatch_thread_id.x / screen_size.x), vec3<f32>(1.0, 1.0, 0.0), false, msaaSampleIndex);
//         return;
// #endif

        let pixel_pos = vec2<u32>((pixel_id >> 18u), pixel_id & 0x3FFFu);
#if CMAA_MSAA_SAMPLE_COUNT > 1
        msaaSampleIndex = (pixel_id >> 14u) & 0x07u;
#endif

// #if CMAA_MSAA_SAMPLE_COUNT > 1
//      let loadPosCenter = vec4<i32>(vec2<i32>(pixel_pos), i32(msaaSampleIndex), 0);
// #else
//      let loadPosCenter = vec3<i32>(vec2<i32>(pixel_pos), 0);
// #endif

        let edges_center_packed = LoadEdge(pixel_pos, vec2<i32>(0, 0), msaaSampleIndex);
        let edges: vec4<f32>      = UnpackEdgesFlt(edges_center_packed);
        let edges_left: vec4<f32>  = UnpackEdgesFlt(LoadEdge(pixel_pos, vec2<i32>(-1, 0), msaaSampleIndex));
        let edges_right: vec4<f32> = UnpackEdgesFlt(LoadEdge(pixel_pos, vec2<i32>(1, 0), msaaSampleIndex));
        let edges_btm: vec4<f32> = UnpackEdgesFlt(LoadEdge(pixel_pos, vec2<i32>(0, 1), msaaSampleIndex));
        let edges_top: vec4<f32>   = UnpackEdgesFlt(LoadEdge(pixel_pos, vec2<i32>(0, -1), msaaSampleIndex));
    
        // simple shapes
        {
            let blend_val = ComputeSimpleShapeBlendValues(edges, edges_left, edges_right, edges_top, edges_btm, true);

            const four_weight_sum = dot(blend_val, vec4<f32>(1.0) );
            const center_weight = 1.0 - four_weight_sum;

            var out_color = LoadSourceColor(pixel_pos, vec2<i32>(0, 0), msaaSampleIndex).rgb * center_weight;
            if( blend_val.x > 0.0 )   // from left
            {
                let pixel_l = LoadSourceColor(pixel_pos, vec2<i32>(-1, 0), msaaSampleIndex).rgb;
                out_color.rgb += blend_val.x * pixel_l;
            }
            if( blend_val.y > 0.0 )   // from above
            {
                let pixel_t = LoadSourceColor(pixel_pos, vec2<i32>(0, -1), msaaSampleIndex).rgb; 
                out_color.rgb += blend_val.y * pixel_t;
            }
            if( blend_val.z > 0.0 )   // from right
            {
                let pixel_r = LoadSourceColor(pixel_pos, vec2<i32>(1, 0), msaaSampleIndex).rgb;
                out_color.rgb += blend_val.z * pixel_r;
            }
            if( blend_val.w > 0.0 )   // from below
            {
                let pixel_b = LoadSourceColor(pixel_pos, vec2<i32>(0, 1), msaaSampleIndex).rgb;
                out_color.rgb += blend_val.w * pixel_b;
            }

            StoreColorSample(pixel_pos.xy, out_color, false, msaaSampleIndex);
        }

        // complex shapes - detect
        {
            var inverted_z_score: f32;
            var normal_z_score: f32;
            var max_score: f32;
            var horizontal = true;
            var inverted_z = false;

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // horizontal
            {
                let edgesM1P0 = edges_left;
                let edgesP1P0 = edges_right;
                let edgesP2P0 = UnpackEdgesFlt(LoadEdge(pixel_pos, vec2<i32>(2, 0), msaaSampleIndex));

                DetectZsHorizontal(edges, edgesM1P0, edgesP1P0, edgesP2P0, &inverted_z_score, &normal_z_score);
                max_score = max( inverted_z_score, normal_z_score );

                if(max_score > 0.0)
                {
                    inverted_z = inverted_z_score > normal_z_score;
                }
            }
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // vertical
            {
                // Reuse the same code for vertical (used for horizontal above), but rotate input data 90 degrees counter-clockwise, so that:
                // left     becomes     bottom
                // top      becomes     left
                // right    becomes     top
                // bottom   becomes     right

                // we also have to rotate edges, thus .argb
                let edgesM1P0 = edges_btm;
                let edgesP1P0 = edges_top;
                let edgesP2P0 = UnpackEdgesFlt(LoadEdge(pixel_pos, vec2<i32>(0, -2), msaaSampleIndex));

                DetectZsHorizontal(edges.argb, edgesM1P0.argb, edgesP1P0.argb, edgesP2P0.argb, &inverted_z_score, &normal_z_score);
                let vert_score = max(inverted_z_score, normal_z_score);

                if( vert_score > max_score )
                {
                    max_score = vert_score;
                    horizontal = false;
                    inverted_z = inverted_z_score > normal_z_score;
                }
            }
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            if(max_score > 0.0)
            {
#ifdef CMAA2_EXTRA_SHARPNESS
                let shape_quality_score = round(clamp(4.0 - max_score, 0.0, 3.0));    // 0 - best quality, 1 - some edges missing but ok, 2 & 3 - dubious but better than nothing
#else
                let shape_quality_score = floor(clamp(4.0 - max_score, 0.0, 3.0));    // 0 - best quality, 1 - some edges missing but ok, 2 & 3 - dubious but better than nothing
#endif

                // const float2 step_right = ( horizontal ) ? ( float2( 1, 0 ) ) : ( float2( 0, -1 ) );
                var step_right: vec2<f32>;
                if(horizontal) {
                    step_right = vec2<f32>(1.0, 0.0);
                } else {
                    step_right = vec2<f32>(0.0, -1.0);
                }
                var line_length_left: f32;
                var line_length_right: f32;
                FindZLineLengths(&line_length_left, &line_length_right, pixel_pos, horizontal, inverted_z, step_right, msaaSampleIndex);

                line_length_left  -= shape_quality_score;
                line_length_right -= shape_quality_score;

                if( ( line_length_left + line_length_right ) >= (5.0) )
                {
#ifdef CMAA2_COLLECT_EXPAND_BLEND_ITEMS
                    // try adding to SLM but fall back to in-place processing if full (which only really happens in synthetic test cases)
                    if(!CollectBlendZs(pixel_pos, horizontal, inverted_z, shape_quality_score, line_length_left, line_length_right, step_right, msaaSampleIndex))
#endif
                    {
                        BlendZs(pixel_pos, horizontal, inverted_z, shape_quality_score, line_length_left, line_length_right, step_right, msaaSampleIndex);
                    }
                }
            }
        }
    }

#ifdef CMAA2_COLLECT_EXPAND_BLEND_ITEMS
    workgroupBarrier();

    let total_item_count: u32 = min(u32(CMAA2_BLEND_ITEM_SLM_SIZE), atomicLoad(&group_shared_blend_item_count));

    // spread items into waves
    let loops: u32 = (total_item_count+(CMAA2_PROCESS_CANDIDATES_NUM_THREADS - 1u) - group_thread_id.x) / CMAA2_PROCESS_CANDIDATES_NUM_THREADS;

    for(var loop_count = 0u; loop_count < loops; loop_count++) {
        let index: u32 = loop_count * CMAA2_PROCESS_CANDIDATES_NUM_THREADS + group_thread_id.x;

        let item_val = group_shared_blend_items[index];

        let starting_pos = vec2<u32>((item_val.x >> 18u), item_val.x & 0x3FFFu);
        var item_msaa_sample_idx = 0u;
#if CMAA_MSAA_SAMPLE_COUNT > 1
        item_msaa_sample_idx     = (item_val.x >> 14u) & 0x07u;
#endif

        let item_horizontal: bool = bool((item_val.y >> 31u) & 1u);
        let item_inverted_z: bool = bool((item_val.y >> 30u) & 1u);
        let item_step_idx = f32((item_val.y >> 20u) & 0x3FFu) - 256.0;
        let item_src_offset = f32((item_val.y >> 10u) & 0x3FFu) - 256.0;
        let item_lerp_k = f32(item_val.y & 0x3FFu) / 1023.0;

        var item_step_right: vec2<f32>;
        var item_blend_dir: vec2<f32>;
        if item_horizontal {
            item_step_right = vec2<f32>(1.0, 0.0);
            item_blend_dir = vec2<f32>(0.0, -1.0);
        } else {
            item_step_right = vec2<f32>(0.0, -1.0);
            item_blend_dir = vec2<f32>(-1.0, 0.0);
        }
        if item_inverted_z {
            item_blend_dir = -item_blend_dir;
        }

        let item_pixel_pos = starting_pos + vec2<u32>(item_step_right * item_step_idx);

        let color_center = LoadSourceColor(item_pixel_pos, vec2<i32>(0), item_msaa_sample_idx).rgb;
        let color_from = LoadSourceColor(item_pixel_pos.xy + vec2<u32>(item_blend_dir * item_src_offset), vec2<i32>(0), item_msaa_sample_idx).rgb;
        
        let output_color = mix(color_center.rgb, color_from.rgb, item_lerp_k);

        StoreColorSample(item_pixel_pos.xy, output_color, true, item_msaa_sample_idx);
    }
#endif
}


#ifdef CMAA2_DEFERRED_APPLY_THREADGROUP_SWAP
@compute @workgroup_size(4, CMAA2_DEFERRED_APPLY_NUM_THREADS, 1)
// [numthreads( 4, CMAA2_DEFERRED_APPLY_NUM_THREADS, 1 )]
#else
@compute @workgroup_size(CMAA2_DEFERRED_APPLY_NUM_THREADS, 4, 1)
// [numthreads( CMAA2_DEFERRED_APPLY_NUM_THREADS, 4, 1 )]
#endif
fn DeferredColorApply2x2CS(@builtin(global_invocation_id) dispatch_thread_id : vec3<u32>, @builtin(local_invocation_id) group_thread_id: vec3<u32>) {
    const num_candidates = atomicLoad(&working_control_buffer[12]);
#ifdef CMAA2_DEFERRED_APPLY_THREADGROUP_SWAP
    const current_candidate = dispatch_thread_id.y;
    const current_quad_offset_xy = group_thread_id.x;
#else
    const current_candidate = dispatch_thread_id.x;
    const current_quad_offset_xy = group_thread_id.y;
#endif

    if current_candidate >= num_candidates {
        return;
    }
    let pixel_id = working_deferred_blend_location_list[current_candidate];
    let quad_pos = vec2<u32>((pixel_id >> 16u), pixel_id & 0xFFFFu);
    const qe_offsets = array<vec2<u32>, 4>(vec2<u32>(0u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u));
    let pixel_pos = quad_pos * 2u + qe_offsets[current_quad_offset_xy];

    var counter_idx_with_header = atomicLoad(&working_deferred_blend_item_list_heads[quad_pos]);

    var counter = 0u;

#if CMAA_MSAA_SAMPLE_COUNT > 1
    var out_colors: array<vec4<f32>, CMAA_MSAA_SAMPLE_COUNT>;
    for(var msaaSampleIndex = 0u; msaaSampleIndex < CMAA_MSAA_SAMPLE_COUNT; msaaSampleIndex++) {
        out_colors[msaaSampleIndex] = vec4<f32>(0.0);
    }
    var has_value = false;
#else
    var out_colors = vec4<f32>(0.0);
#endif

    const max_loops: u32 = 32 * CMAA_MSAA_SAMPLE_COUNT;   // do the loop to prevent bad data hanging the GPU <- probably not needed
    {
        for(var i = 0u; (counter_idx_with_header != 0xFFFFFFFFu) && (i < max_loops); i++) {
            // decode item-specific info: {2 bits for 2x2 quad location}, {3 bits for MSAA sample index}, {1 bit for is_complex_shape flag}, {26 bits for address}
            let offset_xy = (counter_idx_with_header >> 30u) & 0x03u;
            let msaaSampleIndex = (counter_idx_with_header >> 27u) & 0x07u;
            let is_complex_shape = bool((counter_idx_with_header >> 26u) & 0x01u);

            // uint2 val = g_workingDeferredBlendItemList[ counter_idx_with_header & ((1 << 26) - 1) ];
            let val = working_deferred_blend_item_list[counter_idx_with_header & ((1u << 26u) - 1u)];

            counter_idx_with_header  = val.x;

            if offset_xy == current_quad_offset_xy {
                let color = InternalUnpackColor(val.y);
                let weight = 0.8 + 1.0 * f32(is_complex_shape);
#if CMAA_MSAA_SAMPLE_COUNT > 1
                out_colors[msaaSampleIndex] += vec4<f32>(color * weight, weight);
                has_value = true;
#else
                out_colors += vec4<f32>(color * weight, weight);
#endif
            }
        }
    }

#if CMAA_MSAA_SAMPLE_COUNT > 1
    if !has_value {
        return;
    }
#else
    if out_colors.a == 0.0 {
        return;
    }
#endif

    {
#if CMAA_MSAA_SAMPLE_COUNT > 1
        var out_color = vec4<f32>(0.0);
        for(var msaaSampleIndex = 0u; msaaSampleIndex < CMAA_MSAA_SAMPLE_COUNT; msaaSampleIndex++)
        {
            if(out_colors[msaaSampleIndex].a != 0.0) {
                out_color.xyz += out_colors[msaaSampleIndex].rgb / (out_colors[msaaSampleIndex].a);
            }
            else {
                out_color.xyz += LoadSourceColor(pixel_pos, vec2<i32>(0), msaaSampleIndex);
            }
        }
        out_color /= f32(CMAA_MSAA_SAMPLE_COUNT);
#else
        var out_color = out_colors;
        out_color.rgb /= out_color.a;
#endif
        FinalUAVStore(pixel_pos, out_color.rgb);
    }
}

@compute @workgroup_size(16, 16, 1)
fn DebugDrawEdgesCS(@builtin(global_invocation_id) dispatch_thread_id : vec3<u32>)
{
    var msaaSampleIndex = 0u;
    let edges = UnpackEdgesFlt(LoadEdge(dispatch_thread_id.xy, vec2<i32>(0), msaaSampleIndex));

    // show MSAA control mask
    // uint v = g_inColorMSComplexityMaskReadonly.Load( int3( dispatch_thread_id, 0 ) );
    // FinalUAVStore( dispatch_thread_id, float3( v, v, v ) );
    // return;

// #ifdef 0
// #if CMAA_MSAA_SAMPLE_COUNT > 1
//     let pixel_pos = dispatch_thread_id.xy / 2u * 2u;

// #ifdef CMAA_MSAA_USE_COMPLEXITY_MASK
//     float2 texSize;

//     g_inColorMSComplexityMaskReadonly.GetDimensions( texSize.x, texSize.y );
//     float2 gatherUV = vec2<f32>(pixel_pos) / texSize;
//     float4 TL = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gatherUV, int2( 0, 0 ) );
//     float4 TR = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gatherUV, int2( 2, 0 ) );
//     float4 BL = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gatherUV, int2( 0, 2 ) );
//     float4 BR = g_inColorMSComplexityMaskReadonly.GatherRed( g_gather_point_clamp_Sampler, gatherUV, int2( 2, 2 ) );
//     float4 sumAll = TL+TR+BL+BR;
//     bool firstLoopIsEnough = !any(sumAll);

//     //all_2x2_ms_samples_different = (all_2x2_ms_samples_different != 0)?(CMAA_MSAA_SAMPLE_COUNT):(1);
//     FinalUAVStore( dispatch_thread_id, (firstLoopIsEnough).xxx );
//     return;
// #endif
// #endif
// #endif


    //if( any(edges) )
    {
        let output_color = vec4<f32>(mix(edges.xyz, vec3<f32>(0.5), edges.a * 0.2), 1.0);
        FinalUAVStore(dispatch_thread_id.xy, output_color.rgb);
    }

//#if CMAA2_EDGE_DETECTION_LUMA_PATH == 2
//    FinalUAVStore( dispatch_thread_id, g_inLumaReadonly.Load( int3( dispatch_thread_id.xy, 0 ) ).r );
//#endif
}