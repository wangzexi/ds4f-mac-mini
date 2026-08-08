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
 * The <=32-row path mirrors DwarfStar's raw decode FlashAttention exactly:
 * F16 Q/K/V conversion, one SIMD group, four float4 dot partials per lane,
 * and its online softmax/sink ordering.  Longer contexts retain the former
 * probe until the split-workgroup reduction is brought over as well.
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
"  if (nrows <= 32u) {\n"
"    const uint qoff = head * 512u;\n"
"    device const float4 *q4 = (device const float4 *)(q + qoff);\n"
"    device const float4 *kv4 = (device const float4 *)kv;\n"
"    float score = -65504.0f;\n"
"    for (uint cc = 0u; cc < 32u; ++cc) {\n"
"      float partial = 0.0f;\n"
"      if (cc < nrows) {\n"
"        for (uint ii = 0u; ii < 4u; ++ii) {\n"
"          const float4 qq = (float4)(half4)q4[ii * 32u + lane];\n"
"          const float4 kk = (float4)(half4)kv4[cc * 128u + ii * 32u + lane];\n"
"          partial += dot(kk, qq);\n"
"        }\n"
"      }\n"
"      const float dotv = simd_sum(partial);\n"
"      if (cc == lane && cc < nrows) score = dotv * 0.04419417382415922f;\n"
"    }\n"
"    threadgroup float ss[32];\n"
"    ss[lane] = score;\n"
"    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float M = simd_max(max(-FLT_MAX/2.0f, score));\n"
"    const float vs = exp(score - M);\n"
"    float S = simd_sum(vs);\n"
"    ss[lane] = vs;\n"
"    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;\n"
"    for (uint cc = 0u; cc < nrows; ++cc) {\n"
"      const float w = ss[cc];\n"
"      o0 += (float4)(half4)kv4[cc * 128u + 0u * 32u + lane] * w;\n"
"      o1 += (float4)(half4)kv4[cc * 128u + 1u * 32u + lane] * w;\n"
"      o2 += (float4)(half4)kv4[cc * 128u + 2u * 32u + lane] * w;\n"
"      o3 += (float4)(half4)kv4[cc * 128u + 3u * 32u + lane] * w;\n"
"    }\n"
"    const float m = M;\n"
"    const float sink = lane == 0u ? sinks[head] : -FLT_MAX/2.0f;\n"
"    M = simd_max(max(M, sink));\n"
"    const float ms = exp(m - M);\n"
"    const float sinkv = exp(sink - M);\n"
"    S = S * ms + simd_sum(sinkv);\n"
"    o0 *= ms; o1 *= ms; o2 *= ms; o3 *= ms;\n"
"    const float invs = S == 0.0f ? 0.0f : 1.0f / S;\n"
"    ((device float4 *)(out + qoff))[0u * 32u + lane] = o0 * invs;\n"
"    ((device float4 *)(out + qoff))[1u * 32u + lane] = o1 * invs;\n"
"    ((device float4 *)(out + qoff))[2u * 32u + lane] = o2 * invs;\n"
"    ((device float4 *)(out + qoff))[3u * 32u + lane] = o3 * invs;\n"
"    return;\n"
"  }\n"
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
                                const float *comp_kv, const float *sinks,
                                uint32_t n_raw, uint32_t n_comp, float *out) {
    const uint32_t n_keys = n_raw + n_comp;
    if (!q || !raw_kv || !sinks || !out || n_raw == 0 || n_keys > DS4F_ATTN_RAW_MAX ||
        (n_comp && !comp_kv))
        return -1;
    @autoreleasepool {
        if (attention_init() != 0) return -1;
        const NSUInteger q_bytes = DS4F_ATTN_HEADS * DS4F_ATTN_HEAD_DIM * sizeof(float);
        const NSUInteger kv_bytes = (NSUInteger)n_keys * DS4F_ATTN_HEAD_DIM * sizeof(float);
        float *joined_kv = malloc(kv_bytes);
        if (!joined_kv) return -1;
        memcpy(joined_kv, raw_kv, (size_t)n_raw * DS4F_ATTN_HEAD_DIM * sizeof(float));
        if (n_comp) {
            memcpy(joined_kv + (size_t)n_raw * DS4F_ATTN_HEAD_DIM, comp_kv,
                   (size_t)n_comp * DS4F_ATTN_HEAD_DIM * sizeof(float));
        }
        id<MTLBuffer> qb = [g_attention_device newBufferWithBytes:q length:q_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> kvb = [g_attention_device newBufferWithBytes:joined_kv length:kv_bytes
                                                            options:MTLResourceStorageModeShared];
        free(joined_kv);
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
        [enc setBytes:&n_keys length:sizeof(n_keys) atIndex:4];
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
