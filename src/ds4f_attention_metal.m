#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DS4F_ATTN_HEADS = 64, DS4F_ATTN_HEAD_DIM = 512, DS4F_ATTN_RAW_MAX = 128 };

static id<MTLDevice> g_attention_device;
static id<MTLCommandQueue> g_attention_queue;
static id<MTLComputePipelineState> g_attention_pipeline;
static int g_attention_initialized;

/*
 * Short-context raw-KV decode kernel.  The threadgroup layout deliberately
 * follows the model's 64 x 512 head geometry: lane zero forms the softmax
 * scores, then all 32 lanes accumulate independent output dimensions.
 * This is a narrow numerical probe before the full prefill kernel is moved.
 */
static const char *g_attention_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void ds4f_attention_raw(\n"
"    device const float *q [[buffer(0)]],\n"
"    device const float *kv [[buffer(1)]],\n"
"    device const float *sinks [[buffer(2)]],\n"
"    device float *out [[buffer(3)]],\n"
"    constant uint &nrows [[buffer(4)]],\n"
"    constant uint &round_q [[buffer(5)]],\n"
"    uint head [[threadgroup_position_in_grid]],\n"
"    uint lane [[thread_index_in_threadgroup]]) {\n"
"  threadgroup float logits[128];\n"
"  threadgroup float maxv;\n"
"  threadgroup float den;\n"
"  const uint qoff = head * 512u;\n"
"  if (lane == 0u) {\n"
"    float m = sinks[head];\n"
"    for (uint r = 0u; r < nrows; ++r) {\n"
"      float dotv = 0.0f;\n"
"      const uint koff = r * 512u;\n"
"      for (uint j = 0u; j < 512u; ++j) {\n"
"        const float qv = round_q ? float(half(q[qoff + j])) : q[qoff + j];\n"
"        dotv = fma(qv, kv[koff + j], dotv);\n"
"      }\n"
"      const float s = dotv * 0.04419417382415922f;\n"
"      logits[r] = s;\n"
"      m = max(m, s);\n"
"    }\n"
"    maxv = m;\n"
"    float d = exp(sinks[head] - m);\n"
"    for (uint r = 0u; r < nrows; ++r) d += exp(logits[r] - m);\n"
"    den = d;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  for (uint j = lane; j < 512u; j += 32u) {\n"
"    float v = 0.0f;\n"
"    for (uint r = 0u; r < nrows; ++r) {\n"
"      const float w = exp(logits[r] - maxv) / den;\n"
"      v = fma(w, kv[r * 512u + j], v);\n"
"    }\n"
"    out[qoff + j] = v;\n"
"  }\n"
"}\n";

static int attention_init(void) {
    if (g_attention_initialized) return g_attention_pipeline ? 0 : -1;
    g_attention_initialized = 1;
    @autoreleasepool {
        g_attention_device = MTLCreateSystemDefaultDevice();
        if (!g_attention_device) return -1;
        g_attention_queue = [g_attention_device newCommandQueue];
        if (!g_attention_queue) return -1;
        NSError *error = nil;
        /* DwarfStar ships this kernel family with Metal's default fast math. */
        MTLCompileOptions *options = [MTLCompileOptions new];
        NSString *source = [NSString stringWithUTF8String:g_attention_source];
        id<MTLLibrary> library = [g_attention_device newLibraryWithSource:source
                                                                    options:options
                                                                      error:&error];
        if (!library) {
            fprintf(stderr, "ds4f: Metal attention library compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"ds4f_attention_raw"];
        g_attention_pipeline =
            [g_attention_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_attention_pipeline) {
            fprintf(stderr, "ds4f: Metal attention pipeline failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }
    }
    return 0;
}

int ds4f_metal_attention_decode(const float *q, const float *raw_kv,
                                const float *sinks, uint32_t n_raw,
                                float *out) {
    if (!q || !raw_kv || !sinks || !out || n_raw == 0 || n_raw > DS4F_ATTN_RAW_MAX)
        return -1;
    @autoreleasepool {
        if (attention_init() != 0) return -1;
        const NSUInteger q_bytes = DS4F_ATTN_HEADS * DS4F_ATTN_HEAD_DIM * sizeof(float);
        const NSUInteger kv_bytes = (NSUInteger)n_raw * DS4F_ATTN_HEAD_DIM * sizeof(float);
        id<MTLBuffer> qb = [g_attention_device newBufferWithBytes:q length:q_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> kvb = [g_attention_device newBufferWithBytes:raw_kv length:kv_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = [g_attention_device newBufferWithBytes:sinks
                                                            length:DS4F_ATTN_HEADS * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [g_attention_device newBufferWithLength:q_bytes
                                                            options:MTLResourceStorageModeShared];
        if (!qb || !kvb || !sb || !ob) return -1;
        id<MTLCommandBuffer> cb = [g_attention_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_attention_pipeline];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kvb offset:0 atIndex:1];
        [enc setBuffer:sb offset:0 atIndex:2];
        [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBytes:&n_raw length:sizeof(n_raw) atIndex:4];
        const uint32_t round_q = getenv("DS4F_METAL_ATTENTION_F32_Q") == NULL;
        [enc setBytes:&round_q length:sizeof(round_q) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(DS4F_ATTN_HEADS, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        memcpy(out, ob.contents, q_bytes);
    }
    return 0;
}
