#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ds4f_gguf.h"
#include "ds4f_quant.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_pipeline;
static id<MTLComputePipelineState> g_q8_group_pipeline;
static id<MTLComputePipelineState> g_q8_pair_pipeline;
static id<MTLComputePipelineState> g_iq2_pipeline;
static id<MTLComputePipelineState> g_iq2_pair_pipeline;
static id<MTLComputePipelineState> g_iq2_probe_pipeline;
static id<MTLComputePipelineState> g_q2_pipeline;
static id<MTLBuffer> g_iq2_signed_grid;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_buffers;
static NSMutableArray<NSString *> *g_order;
static NSMutableDictionary<NSString *, NSNumber *> *g_buffer_is_expert;
static uint64_t g_cache_bytes;
static uint64_t g_cache_limit;
static uint64_t g_cache_hits;
static uint64_t g_cache_misses;
static uint64_t g_dense_load_bytes;
static uint64_t g_expert_load_bytes;
static uint64_t g_cache_evicted_bytes;
static int g_initialized;

/* Match DwarfStar's `iq2xxs_signed_grid[g][s][j]` literally, keeping the CPU
 * and Metal paths on one representation for IQ2 sign/parity decoding. */
static int32_t g_iq2_signed_grid_data[256u * 128u * 8u];

static void build_iq2_signed_grid(void) {
    const uint64_t *raw = ds4f_iq2_grid_data();
    for (uint32_t g = 0; g < 256u; ++g) {
        const uint8_t *values = (const uint8_t *)(raw + g);
        for (uint32_t code = 0; code < 128u; ++code) {
            uint32_t signs = code | (((uint32_t)__builtin_popcount(code) & 1u) << 7u);
            for (uint32_t j = 0; j < 8u; ++j) {
                int value = values[j];
                if (signs & (1u << j)) value = -value;
                g_iq2_signed_grid_data[(g * 128u + code) * 8u + j] = value;
            }
        }
    }
}

/* DwarfStar's routed-expert Metal kernels consume raw F32 ffn_norm values. */
/* Do not insert a Q8_K quantize/dequantize round-trip before dispatch. */

/* DwarfStar's Metal dense-Q8 kernels consume raw F32 activations. */
/* Keep activation quantization out of the Metal path. */

static const char *g_kernel_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void q8_matvec(device const char *w [[buffer(0)]],\n"
"                       device const float *x [[buffer(1)]],\n"
"                       device float *y [[buffer(2)]],\n"
"                       constant uint &in_dim [[buffer(3)]],\n"
"                       constant uint &row0 [[buffer(4)]],\n"
"                       constant uint &rows [[buffer(5)]],\n"
"                       uint tg [[threadgroup_position_in_grid]],\n"
"                       uint lane [[thread_index_in_threadgroup]]) {\n"
"    const uint first_row = tg * 4u;\n"
"    if (first_row >= rows) return;\n"
"    uint blocks = (in_dim + 31u) / 32u;\n"
"    uint row_bytes = blocks * 34u;\n"
"    threadgroup float partial[4][32];\n"
"    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    for (uint b = lane; b < blocks; b += 32u) {\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            if (first_row + r < rows) {\n"
"                device const char *row = w + (row0 + first_row + r) * row_bytes + b * 34u;\n"
"                ushort hb = (ushort)(uchar)row[0] | ((ushort)(uchar)row[1] << 8);\n"
"                float d = float(as_type<half>(hb));\n"
"                for (uint i = 0; i < 32u; ++i) sums[r] += d * float(int(row[2u + i])) * x[b * 32u + i];\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; ++r) partial[r][lane] = sums[r];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (lane == 0u) {\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            float sum = 0.0f;\n"
"            for (uint i = 0; i < 32u; ++i) sum += partial[r][i];\n"
"            if (first_row + r < rows) y[first_row + r] = sum;\n"
"        }\n"
"    }\n"
"}\n"
"kernel void q8_matvec_group(device const char *w [[buffer(0)]],\n"
"                            device const float *x [[buffer(1)]],\n"
"                            device float *y [[buffer(2)]],\n"
"                            constant uint &group_in [[buffer(3)]],\n"
"                            constant uint &group_out [[buffer(4)]],\n"
"                            constant uint &n_groups [[buffer(5)]],\n"
"                            uint tg [[threadgroup_position_in_grid]],\n"
"                            uint lane [[thread_index_in_threadgroup]]) {\n"
"    const uint chunks = (group_out + 3u) / 4u;\n"
"    const uint group = tg / chunks;\n"
"    const uint local_row = (tg % chunks) * 4u;\n"
"    if (group >= n_groups || local_row >= group_out) return;\n"
"    const uint first_row = group * group_out + local_row;\n"
"    const uint blocks = (group_in + 31u) / 32u;\n"
"    const uint row_bytes = blocks * 34u;\n"
"    device const float *gx = x + group * group_in;\n"
"    threadgroup float partial[4][32];\n"
"    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    for (uint b = lane; b < blocks; b += 32u) {\n"
"        for (uint r = 0u; r < 4u; ++r) if (local_row + r < group_out) {\n"
"            device const char *row = w + (first_row + r) * row_bytes + b * 34u;\n"
"            ushort hb = ushort(uchar(row[0])) | (ushort(uchar(row[1])) << 8u);\n"
"            const float d = float(as_type<half>(hb));\n"
"            for (uint i = 0u; i < 32u; ++i) sums[r] += d * float(int(row[2u + i])) * gx[b * 32u + i];\n"
"        }\n"
"    }\n"
"    for (uint r = 0u; r < 4u; ++r) partial[r][lane] = sums[r];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (lane == 0u) for (uint r = 0u; r < 4u; ++r) if (local_row + r < group_out) {\n"
"        float sum = 0.0f; for (uint i = 0u; i < 32u; ++i) sum += partial[r][i];\n"
"        y[first_row + r] = sum;\n"
"    }\n"
"}\n";

static const char *g_q8_pair_kernel_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void q8_matvec_pair(device const char *wa [[buffer(0)]],\n"
"                           device const char *wb [[buffer(1)]],\n"
"                           device const float *x [[buffer(2)]],\n"
"                           device float *ya [[buffer(3)]],\n"
"                           device float *yb [[buffer(4)]],\n"
"                           constant uint &in_dim [[buffer(5)]],\n"
"                           constant uint &rows [[buffer(6)]],\n"
"                           uint tg [[threadgroup_position_in_grid]],\n"
"                           uint lane [[thread_index_in_threadgroup]]) {\n"
"    const uint first_row = tg * 4u;\n"
"    if (first_row >= rows) return;\n"
"    const uint blocks = (in_dim + 31u) / 32u;\n"
"    const uint row_bytes = blocks * 34u;\n"
"    threadgroup float pa[4][32];\n"
"    threadgroup float pb[4][32];\n"
"    float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    float sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    for (uint b = lane; b < blocks; b += 32u) {\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            if (first_row + r < rows) {\n"
"                device const char *ra = wa + (first_row + r) * row_bytes + b * 34u;\n"
"                device const char *rb = wb + (first_row + r) * row_bytes + b * 34u;\n"
"                ushort ha = (ushort)(uchar)ra[0] | ((ushort)(uchar)ra[1] << 8);\n"
"                ushort hb = (ushort)(uchar)rb[0] | ((ushort)(uchar)rb[1] << 8);\n"
"                float za = 0.0f, zb = 0.0f;\n"
"                for (uint i = 0; i < 32u; ++i) {\n"
"                    float v = x[b * 32u + i];\n"
"                    za += float(int(ra[2u + i])) * v;\n"
"                    zb += float(int(rb[2u + i])) * v;\n"
"                }\n"
"                sa[r] += float(as_type<half>(ha)) * za;\n"
"                sb[r] += float(as_type<half>(hb)) * zb;\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; ++r) { pa[r][lane] = sa[r]; pb[r][lane] = sb[r]; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (lane == 0u) {\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            float suma = 0.0f, sumb = 0.0f;\n"
"            for (uint i = 0; i < 32u; ++i) { suma += pa[r][i]; sumb += pb[r][i]; }\n"
"            if (first_row + r < rows) { ya[first_row + r] = suma; yb[first_row + r] = sumb; }\n"
"        }\n"
"    }\n"
"}\n";

static const char *g_expert_kernel_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void iq2_expert(device const char *w [[buffer(0)]],\n"
"                        device const float *x [[buffer(1)]],\n"
"                        device const int *signed_grid [[buffer(2)]],\n"
"                        device float *y [[buffer(3)]],\n"
"                        constant uint &in_dim [[buffer(4)]],\n"
"                        constant uint &rows [[buffer(5)]],\n"
"                        uint tg [[threadgroup_position_in_grid]],\n"
"                        uint lane [[thread_index_in_threadgroup]]) {\n"
"    const uint rows_per_group = 4u;\n"
"    const uint first_row = tg * rows_per_group;\n"
"    if (first_row >= rows) return;\n"
"    const uint blocks = in_dim / 256u;\n"
"    const uint row_bytes = 66u * blocks;\n"
"    threadgroup float partial[4][32];\n"
"    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    const uint group = lane & 7u;\n"
"    for (uint b = lane >> 3u; b < blocks; b += 4u) {\n"
"        float xv[32];\n"
"        for (uint i = 0; i < 32u; ++i) xv[i] = x[b * 256u + group * 32u + i];\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            if (first_row + r < rows) {\n"
"                device const uchar *rw = (device const uchar *)w +\n"
"                    (first_row + r) * row_bytes + b * 66u + 2u + group * 8u;\n"
"                uint rg = uint(rw[0]) | (uint(rw[1]) << 8u) |\n"
"                          (uint(rw[2]) << 16u) | (uint(rw[3]) << 24u);\n"
"                uint rs = uint(rw[4]) | (uint(rw[5]) << 8u) |\n"
"                          (uint(rw[6]) << 16u) | (uint(rw[7]) << 24u);\n"
"                float rd = float(as_type<half>((ushort)(uint(uchar(w[(first_row + r) * row_bytes + b * 66u])) |\n"
"                                                       (uint(uchar(w[(first_row + r) * row_bytes + b * 66u + 1u])) << 8u))));\n"
"                float rscale = 0.125f * rd * float(2u * (rs >> 28u) + 1u);\n"
"                for (uint l = 0; l < 4u; ++l) {\n"
"                    uint rgi = (rg >> (8u * l)) & 255u;\n"
"                    uint sign_code = (rs >> (7u * l)) & 127u;\n"
"                    for (uint j = 0; j < 8u; ++j) {\n"
"                        float rv = float(signed_grid[(rgi * 128u + sign_code) * 8u + j]);\n"
"                        sums[r] += rscale * rv * xv[l * 8u + j];\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; ++r) partial[r][lane] = sums[r];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (lane == 0u) {\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            float sum = 0.0f;\n"
"            for (uint i = 0; i < 32u; ++i) sum += partial[r][i];\n"
"            if (first_row + r < rows) y[first_row + r] = sum;\n"
"        }\n"
"    }\n"
"}\n"
"kernel void iq2_probe(device const char *w [[buffer(0)]],\n"
"                      device const float *x [[buffer(1)]],\n"
"                      device const int *signed_grid [[buffer(2)]],\n"
"                      device float *y [[buffer(3)]],\n"
"                      constant uint &in_dim [[buffer(4)]],\n"
"                      uint gid [[thread_position_in_grid]]) {\n"
"    if (gid != 0u) return;\n"
"    const uint blocks = in_dim / 256u;\n"
"    float sum = 0.0f;\n"
"    for (uint b = 0; b < blocks; ++b) {\n"
"        ushort hb = ushort(uchar(w[b * 66u])) | (ushort(uchar(w[b * 66u + 1u])) << 8u);\n"
"        float rd = float(as_type<half>(hb));\n"
"        for (uint group = 0; group < 8u; ++group) {\n"
"            device const uchar *rw = (device const uchar *)w + b * 66u + 2u + group * 8u;\n"
"            uint rg = uint(rw[0]) | (uint(rw[1]) << 8u) |\n"
"                      (uint(rw[2]) << 16u) | (uint(rw[3]) << 24u);\n"
"            uint rs = uint(rw[4]) | (uint(rw[5]) << 8u) |\n"
"                      (uint(rw[6]) << 16u) | (uint(rw[7]) << 24u);\n"
"            float scale = 0.125f * rd * float(2u * (rs >> 28u) + 1u);\n"
"            for (uint l = 0; l < 4u; ++l) {\n"
"                uint rgi = (rg >> (8u * l)) & 255u;\n"
"                uint sign_code = (rs >> (7u * l)) & 127u;\n"
"                for (uint j = 0; j < 8u; ++j) {\n"
"                    float rv = float(signed_grid[(rgi * 128u + sign_code) * 8u + j]);\n"
"                    sum += scale * rv * x[b * 256u + group * 32u + l * 8u + j];\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    y[0] = sum;\n"
"}\n"
"kernel void iq2_pair(device const char *wa [[buffer(0)]],\n"
"                     device const char *wb [[buffer(1)]],\n"
"                     device const float *x [[buffer(2)]],\n"
"                     device const int *signed_grid [[buffer(3)]],\n"
"                     device float *ya [[buffer(4)]],\n"
"                     device float *yb [[buffer(5)]],\n"
"                     constant uint &in_dim [[buffer(6)]],\n"
"                     constant uint &rows [[buffer(7)]],\n"
"                     uint tg [[threadgroup_position_in_grid]],\n"
"                     uint lane [[thread_index_in_threadgroup]]) {\n"
"    const uint first_row = tg * 4u;\n"
"    if (first_row >= rows) return;\n"
"    const uint blocks = in_dim / 256u;\n"
"    const uint row_bytes = 66u * blocks;\n"
"    threadgroup float pa[4][32];\n"
"    threadgroup float pb[4][32];\n"
"    float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    float sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    const uint group = lane & 7u;\n"
"    for (uint b = lane >> 3u; b < blocks; b += 4u) {\n"
"        float xv[32];\n"
"        for (uint i = 0; i < 32u; ++i) xv[i] = x[b * 256u + group * 32u + i];\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            if (first_row + r < rows) {\n"
"                device const uchar *rwa = (device const uchar *)wa +\n"
"                    (first_row + r) * row_bytes + b * 66u + 2u + group * 8u;\n"
"                device const uchar *rwb = (device const uchar *)wb +\n"
"                    (first_row + r) * row_bytes + b * 66u + 2u + group * 8u;\n"
"                uint aga = uint(rwa[0]) | (uint(rwa[1]) << 8u) |\n"
"                           (uint(rwa[2]) << 16u) | (uint(rwa[3]) << 24u);\n"
"                uint asa = uint(rwa[4]) | (uint(rwa[5]) << 8u) |\n"
"                           (uint(rwa[6]) << 16u) | (uint(rwa[7]) << 24u);\n"
"                uint agb = uint(rwb[0]) | (uint(rwb[1]) << 8u) |\n"
"                           (uint(rwb[2]) << 16u) | (uint(rwb[3]) << 24u);\n"
"                uint asb = uint(rwb[4]) | (uint(rwb[5]) << 8u) |\n"
"                           (uint(rwb[6]) << 16u) | (uint(rwb[7]) << 24u);\n"
"                float da = float(as_type<half>((ushort)(uint(uchar(wa[(first_row + r) * row_bytes + b * 66u])) |\n"
"                                                         (uint(uchar(wa[(first_row + r) * row_bytes + b * 66u + 1u])) << 8u))));\n"
"                float db = float(as_type<half>((ushort)(uint(uchar(wb[(first_row + r) * row_bytes + b * 66u])) |\n"
"                                                         (uint(uchar(wb[(first_row + r) * row_bytes + b * 66u + 1u])) << 8u))));\n"
"                float scale_a = 0.125f * da * float(2u * (asa >> 28u) + 1u);\n"
"                float scale_b = 0.125f * db * float(2u * (asb >> 28u) + 1u);\n"
"                for (uint l = 0; l < 4u; ++l) {\n"
"                    uint gia = (aga >> (8u * l)) & 255u;\n"
"                    uint gib = (agb >> (8u * l)) & 255u;\n"
"                    uint signa = (asa >> (7u * l)) & 127u;\n"
"                    uint signb = (asb >> (7u * l)) & 127u;\n"
"                    for (uint j = 0; j < 8u; ++j) {\n"
"                        float va = float(signed_grid[(gia * 128u + signa) * 8u + j]);\n"
"                        float vb = float(signed_grid[(gib * 128u + signb) * 8u + j]);\n"
"                        sa[r] += scale_a * va * xv[l * 8u + j];\n"
"                        sb[r] += scale_b * vb * xv[l * 8u + j];\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; ++r) { pa[r][lane] = sa[r]; pb[r][lane] = sb[r]; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (lane == 0u) {\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            float suma = 0.0f, sumb = 0.0f;\n"
"            for (uint i = 0; i < 32u; ++i) { suma += pa[r][i]; sumb += pb[r][i]; }\n"
"            if (first_row + r < rows) { ya[first_row + r] = suma; yb[first_row + r] = sumb; }\n"
"        }\n"
"    }\n"
"}\n"
"kernel void q2_expert(device const char *w [[buffer(0)]],\n"
"                       device const float *x [[buffer(1)]],\n"
"                       device float *y [[buffer(3)]],\n"
"                       constant uint &in_dim [[buffer(4)]],\n"
"                       constant uint &rows [[buffer(5)]],\n"
"                       uint tg [[threadgroup_position_in_grid]],\n"
"                       uint lane [[thread_index_in_threadgroup]]) {\n"
"    const uint rows_per_group = 4u;\n"
"    const uint first_row = tg * rows_per_group;\n"
"    if (first_row >= rows) return;\n"
"    const uint blocks = in_dim / 256u;\n"
"    const uint row_bytes = 84u * blocks;\n"
"    threadgroup float partial[4][32];\n"
"    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    for (uint b = 0; b < blocks; ++b) {\n"
"        for (uint k0 = lane * 8u; k0 < 256u; k0 += 256u) {\n"
"            float xv[8];\n"
"            for (uint j = 0; j < 8u; ++j) xv[j] = x[b * 256u + k0 + j];\n"
"            uint group = k0 / 16u, l0 = k0 & 15u;\n"
"            uint base = 32u * (group / 8u) + 16u * (group & 1u);\n"
"            uint shift = ((group / 2u) & 3u) * 2u;\n"
"            for (uint r = 0; r < 4u; ++r) {\n"
"                if (first_row + r < rows) {\n"
"                    device const uchar *blk = (device const uchar *)w +\n"
"                        (first_row + r) * row_bytes + b * 84u;\n"
"                    ushort db = ushort(blk[80u]) | (ushort(blk[81u]) << 8u);\n"
"                    ushort dmb = ushort(blk[82u]) | (ushort(blk[83u]) << 8u);\n"
"                    float d = float(as_type<half>(db));\n"
"                    float dm = float(as_type<half>(dmb));\n"
"                    for (uint j = 0; j < 8u; ++j) {\n"
"                        uint l = l0 + j;\n"
"                        uint qv = (uint(blk[16u + base + l]) >> shift) & 3u;\n"
"                        uint sc = uint(blk[group]);\n"
"                        float v = d * float(sc & 15u) * float(qv) - dm * float(sc >> 4u);\n"
"                        sums[r] += v * xv[j];\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; ++r) partial[r][lane] = sums[r];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (lane == 0u) {\n"
"        for (uint r = 0; r < 4u; ++r) {\n"
"            float sum = 0.0f;\n"
"            for (uint i = 0; i < 32u; ++i) sum += partial[r][i];\n"
"            if (first_row + r < rows) y[first_row + r] = sum;\n"
"        }\n"
"    }\n"
"}\n";

static uint64_t cache_limit_from_env(void) {
    const char *s = getenv("DS4F_METAL_CACHE_GIB");
    double gib = s && s[0] ? strtod(s, NULL) : 10.0;
    if (!(gib > 0.5)) gib = 0.5;
    if (gib > 13.0) gib = 13.0;
    return (uint64_t)(gib * 1024.0 * 1024.0 * 1024.0);
}

static int metal_init(void) {
    if (g_initialized) return g_pipeline && g_q8_group_pipeline && g_q8_pair_pipeline && g_iq2_pipeline && g_iq2_pair_pipeline && g_iq2_probe_pipeline &&
        g_q2_pipeline && g_iq2_signed_grid ? 0 : -1;
    g_initialized = 1;
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return -1;
        g_queue = [g_device newCommandQueue];
        g_buffers = [NSMutableDictionary dictionary];
        g_order = [NSMutableArray array];
        g_buffer_is_expert = [NSMutableDictionary dictionary];
        g_cache_limit = cache_limit_from_env();
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        /* DwarfStar's production graph uses default fast math.  Keep strict
         * IEEE mode available for diagnostics, but do not make it the default
         * oracle path. */
        if (getenv("DS4F_METAL_MATH_SAFE"))
            options.mathMode = MTLMathModeSafe;
        NSString *source = [NSString stringWithUTF8String:g_kernel_source];
        id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "ds4f: Metal library compile failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"q8_matvec"];
        g_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_pipeline) {
            fprintf(stderr, "ds4f: Metal pipeline failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        id<MTLFunction> group_fn = [library newFunctionWithName:@"q8_matvec_group"];
        g_q8_group_pipeline =
            [g_device newComputePipelineStateWithFunction:group_fn error:&error];
        if (!g_q8_group_pipeline) {
            fprintf(stderr, "ds4f: Metal Q8 group pipeline failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }
        NSString *pair_source = [NSString stringWithUTF8String:g_q8_pair_kernel_source];
        id<MTLLibrary> pair_library = [g_device newLibraryWithSource:pair_source options:options error:&error];
        if (!pair_library) {
            fprintf(stderr, "ds4f: Metal Q8 pair library compile failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        id<MTLFunction> pair_fn = [pair_library newFunctionWithName:@"q8_matvec_pair"];
        g_q8_pair_pipeline = [g_device newComputePipelineStateWithFunction:pair_fn error:&error];
        if (!g_q8_pair_pipeline) {
            fprintf(stderr, "ds4f: Metal Q8 pair pipeline failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        NSString *expert_source = [NSString stringWithUTF8String:g_expert_kernel_source];
        id<MTLLibrary> expert_library = [g_device newLibraryWithSource:expert_source
                                                                  options:options
                                                                    error:&error];
        if (!expert_library) {
            fprintf(stderr, "ds4f: Metal expert library compile failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        id<MTLFunction> iq2_fn = [expert_library newFunctionWithName:@"iq2_expert"];
        id<MTLFunction> iq2_pair_fn = [expert_library newFunctionWithName:@"iq2_pair"];
        id<MTLFunction> iq2_probe_fn = [expert_library newFunctionWithName:@"iq2_probe"];
        id<MTLFunction> q2_fn = [expert_library newFunctionWithName:@"q2_expert"];
        g_iq2_pipeline = [g_device newComputePipelineStateWithFunction:iq2_fn error:&error];
        if (!g_iq2_pipeline) {
            fprintf(stderr, "ds4f: Metal IQ2 pipeline failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        g_iq2_pair_pipeline = [g_device newComputePipelineStateWithFunction:iq2_pair_fn error:&error];
        if (!g_iq2_pair_pipeline) {
            fprintf(stderr, "ds4f: Metal IQ2 pair pipeline failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        g_iq2_probe_pipeline = [g_device newComputePipelineStateWithFunction:iq2_probe_fn error:&error];
        if (!g_iq2_probe_pipeline) {
            fprintf(stderr, "ds4f: Metal IQ2 probe pipeline failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        g_q2_pipeline = [g_device newComputePipelineStateWithFunction:q2_fn error:&error];
        if (!g_q2_pipeline) {
            fprintf(stderr, "ds4f: Metal Q2 pipeline failed: %s\n", error.localizedDescription.UTF8String);
            return -1;
        }
        build_iq2_signed_grid();
        g_iq2_signed_grid = [g_device newBufferWithBytes:g_iq2_signed_grid_data
                                                     length:sizeof(g_iq2_signed_grid_data)
                                                   options:MTLResourceStorageModeShared];
        if (!g_iq2_signed_grid) return -1;
        fprintf(stderr, "ds4f: Metal Q8 backend enabled, cache limit %.2f GiB\n",
                (double)g_cache_limit / (1024.0 * 1024.0 * 1024.0));
    }
    return g_pipeline && g_buffer_is_expert ? 0 : -1;
}

static NSString *buffer_key(const ds4f_gguf *g, const ds4f_tensor *t) {
    return [NSString stringWithFormat:@"%d:%llu:%llu", g->fd,
            (unsigned long long)t->file_offset, (unsigned long long)t->nbytes];
}

__attribute__((destructor))
static void report_cache_stats(void) {
    if (!getenv("DS4F_METAL_CACHE_STATS")) return;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    fprintf(stderr,
            "ds4f: Metal cache stats hits=%llu misses=%llu dense_load=%.2fGiB "
            "expert_load=%.2fGiB evicted=%.2fGiB resident=%.2fGiB\n",
            (unsigned long long)g_cache_hits,
            (unsigned long long)g_cache_misses,
            (double)g_dense_load_bytes / gib,
            (double)g_expert_load_bytes / gib,
            (double)g_cache_evicted_bytes / gib,
            (double)g_cache_bytes / gib);
}

static id<MTLBuffer> get_weight(const ds4f_gguf *g, const ds4f_tensor *t) {
    NSString *key = buffer_key(g, t);
    id<MTLBuffer> existing = g_buffers[key];
    if (existing) {
        ++g_cache_hits;
        return existing;
    }
    if (t->nbytes > g_cache_limit || t->nbytes > SIZE_MAX) return nil;
    ++g_cache_misses;
    if (t->n_dims == 3) g_expert_load_bytes += t->nbytes;
    else g_dense_load_bytes += t->nbytes;
    void *raw = NULL;
    if (posix_memalign(&raw, 16384, (size_t)t->nbytes) != 0 || !raw) return nil;
    if (ds4f_gguf_read(g, t, 0, raw, t->nbytes)) { free(raw); return nil; }
    while (g_cache_bytes + t->nbytes > g_cache_limit && g_order.count) {
        NSUInteger old_index = NSNotFound;
        for (NSUInteger i = 0; i < g_order.count; ++i) {
            NSString *candidate = g_order[i];
            if (g_buffer_is_expert[candidate].boolValue) {
                old_index = i;
                break;
            }
        }
        if (old_index == NSNotFound) old_index = 0;
        NSString *old_key = g_order[old_index];
        id<MTLBuffer> old = g_buffers[old_key];
        g_cache_bytes -= old.length;
        g_cache_evicted_bytes += old.length;
        [g_buffers removeObjectForKey:old_key];
        [g_buffer_is_expert removeObjectForKey:old_key];
        [g_order removeObjectAtIndex:old_index];
        (void)old;
    }
    id<MTLBuffer> buffer = nil;
    if (getenv("DS4F_METAL_FORCE_COPY") == NULL) {
        /* The payload is 16 KiB-aligned and remains owned by the MTLBuffer
         * until LRU eviction.  This avoids a second copy for SSD-streamed
         * weight slices on Apple unified memory. */
        buffer = [g_device newBufferWithBytesNoCopy:raw
                                              length:(NSUInteger)t->nbytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:^(void *pointer, NSUInteger length) {
                                             (void)length;
                                             free(pointer);
                                         }];
        if (buffer) raw = NULL;
    }
    if (!buffer) {
        buffer = [g_device newBufferWithBytes:raw
                                        length:(NSUInteger)t->nbytes
                                       options:MTLResourceStorageModeShared];
    }
    free(raw);
    if (!buffer) return nil;
    if (getenv("DS4F_METAL_NO_CACHE")) return buffer;
    g_buffers[key] = buffer;
    g_buffer_is_expert[key] = @(t->n_dims == 3);
    [g_order addObject:key];
    g_cache_bytes += t->nbytes;
    return buffer;
}

int ds4f_metal_matvec_rows(const ds4f_gguf *g, const ds4f_tensor *t,
                           const float *x, float *y,
                           uint32_t row0, uint32_t rows) {
    if (!g || !t || t->type != 8 || t->n_dims != 2 || !x || !y || !rows) return -1;
    @autoreleasepool {
        if (metal_init() != 0) return -1;
        size_t in = (size_t)t->dims[0];
        id<MTLBuffer> weight = get_weight(g, t);
        id<MTLBuffer> input = [g_device newBufferWithLength:in * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [g_device newBufferWithLength:(NSUInteger)rows * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        if (!weight || !input || !out) return -1;
        memcpy([input contents], x, in * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pipeline];
        [enc setBuffer:weight offset:0 atIndex:0];
        [enc setBuffer:input offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        uint32_t in32 = (uint32_t)in;
        [enc setBytes:&in32 length:sizeof(in32) atIndex:3];
        [enc setBytes:&row0 length:sizeof(row0) atIndex:4];
        [enc setBytes:&rows length:sizeof(rows) atIndex:5];
        NSUInteger groups = (rows + 3) / 4;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        memcpy(y, [out contents], (size_t)rows * sizeof(float));
    }
    return 0;
}

int ds4f_metal_matvec_group(const ds4f_gguf *g, const ds4f_tensor *t,
                            const float *x, uint32_t groups,
                            uint32_t group_in, uint32_t group_out,
                            float *y) {
    if (!g || !t || !x || !y || !groups || !group_in || !group_out ||
        t->type != 8 || t->n_dims != 2 || t->dims[0] != group_in ||
        t->dims[1] != (uint64_t)groups * group_out ||
        (size_t)groups > SIZE_MAX / group_in ||
        (size_t)groups > SIZE_MAX / group_out) return -1;
    @autoreleasepool {
        if (metal_init() != 0 || !g_q8_group_pipeline) return -1;
        const size_t in_floats = (size_t)groups * group_in;
        const size_t out_floats = (size_t)groups * group_out;
        id<MTLBuffer> weight = get_weight(g, t);
        id<MTLBuffer> input =
            [g_device newBufferWithLength:in_floats * sizeof(float)
                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> out =
            [g_device newBufferWithLength:out_floats * sizeof(float)
                                  options:MTLResourceStorageModeShared];
        if (!weight || !input || !out) return -1;
        memcpy(input.contents, x, in_floats * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_q8_group_pipeline];
        [enc setBuffer:weight offset:0 atIndex:0];
        [enc setBuffer:input offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        [enc setBytes:&group_in length:sizeof(group_in) atIndex:3];
        [enc setBytes:&group_out length:sizeof(group_out) atIndex:4];
        [enc setBytes:&groups length:sizeof(groups) atIndex:5];
        const NSUInteger chunks = ((NSUInteger)group_out + 3u) / 4u;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups * chunks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        memcpy(y, out.contents, out_floats * sizeof(float));
    }
    return 0;
}

int ds4f_metal_matvec_q8_pair(const ds4f_gguf *g,
                               const ds4f_tensor *a, const ds4f_tensor *b,
                               const float *x, float *ya, float *yb) {
    if (!g || !a || !b || !x || !ya || !yb || a->type != 8 || b->type != 8 ||
        a->n_dims != 2 || b->n_dims != 2 || a->dims[0] != b->dims[0] ||
        a->dims[1] != b->dims[1] || a->dims[1] > UINT32_MAX) return -1;
    @autoreleasepool {
        if (metal_init() != 0 || !g_q8_pair_pipeline) return -1;
        const size_t in = (size_t)a->dims[0];
        const size_t rows = (size_t)a->dims[1];
        id<MTLBuffer> wa = get_weight(g, a);
        id<MTLBuffer> wb = get_weight(g, b);
        id<MTLBuffer> input = [g_device newBufferWithLength:in * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outa = [g_device newBufferWithLength:rows * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> outb = [g_device newBufferWithLength:rows * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        if (!wa || !wb || !input || !outa || !outb) return -1;
        memcpy([input contents], x, in * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_q8_pair_pipeline];
        [enc setBuffer:wa offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:input offset:0 atIndex:2];
        [enc setBuffer:outa offset:0 atIndex:3];
        [enc setBuffer:outb offset:0 atIndex:4];
        uint32_t in32 = (uint32_t)in, rows32 = (uint32_t)rows;
        [enc setBytes:&in32 length:sizeof(in32) atIndex:5];
        [enc setBytes:&rows32 length:sizeof(rows32) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((rows + 3u) / 4u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        memcpy(ya, [outa contents], rows * sizeof(float));
        memcpy(yb, [outb contents], rows * sizeof(float));
    }
    return 0;
}

static int metal_expert_rows_batch(const ds4f_gguf *g, const ds4f_tensor *t,
                                   const uint32_t *experts, size_t count,
                                   const float *x, size_t x_stride, size_t in,
                                   float *y, size_t y_stride,
                                   id<MTLComputePipelineState> pipeline,
                                   int needs_grid) {
    if (!g || !t || !experts || !count || !x || !y || t->n_dims != 3 ||
        t->dims[2] == 0 || in != t->dims[0] || t->dims[1] > UINT32_MAX ||
        t->nbytes % t->dims[2] || (x_stride && x_stride < in) ||
        y_stride < t->dims[1]) return -1;
    @autoreleasepool {
        if (metal_init() != 0) return -1;
        if (!pipeline) return -1;
        size_t rows = (size_t)t->dims[1];
        uint64_t expert_bytes = t->nbytes / t->dims[2];
        if (expert_bytes > SIZE_MAX) return -1;
        size_t input_floats = x_stride ? (count - 1) * x_stride + in : in;
        id<MTLBuffer> input = [g_device newBufferWithLength:input_floats * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [g_device newBufferWithLength:count * rows * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        if (!input || !out) return -1;
        memcpy([input contents], x, input_floats * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        uint32_t in32 = (uint32_t)in, rows32 = (uint32_t)rows;
        const NSUInteger tg = 32;
        const NSUInteger groups = (rows + 3) / 4;
        for (size_t i = 0; i < count; ++i) {
            ds4f_tensor slice = *t;
            slice.file_offset += (uint64_t)experts[i] * expert_bytes;
            slice.nbytes = expert_bytes;
            id<MTLBuffer> weight = get_weight(g, &slice);
            if (!weight) return -1;
            [enc setBuffer:weight offset:0 atIndex:0];
            NSUInteger input_offset = (NSUInteger)(x_stride ? i * x_stride : 0) * sizeof(float);
            [enc setBuffer:input offset:input_offset atIndex:1];
            if (needs_grid) [enc setBuffer:g_iq2_signed_grid offset:0 atIndex:2];
            [enc setBuffer:out offset:i * rows * sizeof(float) atIndex:3];
            [enc setBytes:&in32 length:sizeof(in32) atIndex:4];
            [enc setBytes:&rows32 length:sizeof(rows32) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        const float *out_data = (const float *)[out contents];
        for (size_t i = 0; i < count; ++i)
            memcpy(y + i * y_stride, out_data + i * rows, rows * sizeof(float));
    }
    return 0;
}

static int metal_expert_rows_pair(const ds4f_gguf *g,
                                  const ds4f_tensor *a,
                                  const ds4f_tensor *b,
                                  const uint32_t *experts, size_t count,
                                  const float *x, size_t x_stride, size_t in,
                                  float *ya, size_t ya_stride,
                                  float *yb, size_t yb_stride,
                                  id<MTLComputePipelineState> pipeline,
                                  int needs_grid) {
    if (!g || !a || !b || !experts || !count || !x || !ya || !yb ||
        a->n_dims != 3 || b->n_dims != 3 || a->type != b->type ||
        a->dims[0] != b->dims[0] || a->dims[2] != b->dims[2] ||
        in != a->dims[0] || a->dims[1] > UINT32_MAX || b->dims[1] > UINT32_MAX ||
        a->nbytes % a->dims[2] || b->nbytes % b->dims[2] ||
        (x_stride && x_stride < in) || ya_stride < a->dims[1] ||
        yb_stride < b->dims[1]) return -1;
    @autoreleasepool {
        if (metal_init() != 0 || !pipeline) return -1;
        size_t rows_a = (size_t)a->dims[1], rows_b = (size_t)b->dims[1];
        uint64_t expert_bytes_a = a->nbytes / a->dims[2];
        uint64_t expert_bytes_b = b->nbytes / b->dims[2];
        size_t input_floats = x_stride ? (count - 1) * x_stride + in : in;
        id<MTLBuffer> input = [g_device newBufferWithLength:input_floats * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_a = [g_device newBufferWithLength:count * rows_a * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_b = [g_device newBufferWithLength:count * rows_b * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        if (!input || !out_a || !out_b) return -1;
        memcpy([input contents], x, input_floats * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        uint32_t in32 = (uint32_t)in, rows_a32 = (uint32_t)rows_a;
        uint32_t rows_b32 = (uint32_t)rows_b;
        const NSUInteger tg = 32;
        const NSUInteger groups_a = (rows_a + 3) / 4;
        const NSUInteger groups_b = (rows_b + 3) / 4;
        for (size_t i = 0; i < count; ++i) {
            if (experts[i] >= a->dims[2]) return -1;
            NSUInteger input_offset = (NSUInteger)(x_stride ? i * x_stride : 0) * sizeof(float);
            ds4f_tensor slice_a = *a;
            slice_a.file_offset += (uint64_t)experts[i] * expert_bytes_a;
            slice_a.nbytes = expert_bytes_a;
            id<MTLBuffer> weight_a = get_weight(g, &slice_a);
            if (!weight_a) return -1;
            id<MTLComputeCommandEncoder> enc_a = [cb computeCommandEncoder];
            [enc_a setComputePipelineState:pipeline];
            [enc_a setBuffer:weight_a offset:0 atIndex:0];
            [enc_a setBuffer:input offset:input_offset atIndex:1];
            if (needs_grid) [enc_a setBuffer:g_iq2_signed_grid offset:0 atIndex:2];
            [enc_a setBuffer:out_a offset:i * rows_a * sizeof(float) atIndex:3];
            [enc_a setBytes:&in32 length:sizeof(in32) atIndex:4];
            [enc_a setBytes:&rows_a32 length:sizeof(rows_a32) atIndex:5];
            [enc_a dispatchThreadgroups:MTLSizeMake(groups_a, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc_a endEncoding];

            ds4f_tensor slice_b = *b;
            slice_b.file_offset += (uint64_t)experts[i] * expert_bytes_b;
            slice_b.nbytes = expert_bytes_b;
            id<MTLBuffer> weight_b = get_weight(g, &slice_b);
            if (!weight_b) return -1;
            id<MTLComputeCommandEncoder> enc_b = [cb computeCommandEncoder];
            [enc_b setComputePipelineState:pipeline];
            [enc_b setBuffer:weight_b offset:0 atIndex:0];
            [enc_b setBuffer:input offset:input_offset atIndex:1];
            if (needs_grid) [enc_b setBuffer:g_iq2_signed_grid offset:0 atIndex:2];
            [enc_b setBuffer:out_b offset:i * rows_b * sizeof(float) atIndex:3];
            [enc_b setBytes:&in32 length:sizeof(in32) atIndex:4];
            [enc_b setBytes:&rows_b32 length:sizeof(rows_b32) atIndex:5];
            [enc_b dispatchThreadgroups:MTLSizeMake(groups_b, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc_b endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        const float *pa = (const float *)[out_a contents];
        const float *pb = (const float *)[out_b contents];
        for (size_t i = 0; i < count; ++i) {
            memcpy(ya + i * ya_stride, pa + i * rows_a, rows_a * sizeof(float));
            memcpy(yb + i * yb_stride, pb + i * rows_b, rows_b * sizeof(float));
        }
    }
    return 0;
}

static int metal_iq2_rows_pair(const ds4f_gguf *g,
                               const ds4f_tensor *a, const ds4f_tensor *b,
                               const uint32_t *experts, size_t count,
                               const float *x, size_t x_stride, size_t in,
                               float *ya, size_t ya_stride,
                               float *yb, size_t yb_stride) {
    if (!g || !a || !b || !experts || !count || !x || !ya || !yb ||
        a->type != 16 || b->type != 16 || a->n_dims != 3 || b->n_dims != 3 ||
        a->dims[0] != b->dims[0] || a->dims[1] != b->dims[1] ||
        a->dims[2] != b->dims[2] || in != a->dims[0] ||
        (x_stride && x_stride < in) || ya_stride < a->dims[1] ||
        yb_stride < b->dims[1] || a->nbytes % a->dims[2] ||
        b->nbytes % b->dims[2]) return -1;
    @autoreleasepool {
        if (metal_init() != 0 || !g_iq2_pair_pipeline) return -1;
        size_t rows = (size_t)a->dims[1];
        uint64_t expert_bytes_a = a->nbytes / a->dims[2];
        uint64_t expert_bytes_b = b->nbytes / b->dims[2];
        size_t input_floats = x_stride ? (count - 1) * x_stride + in : in;
        id<MTLBuffer> input = [g_device newBufferWithLength:input_floats * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_a = [g_device newBufferWithLength:count * rows * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_b = [g_device newBufferWithLength:count * rows * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        if (!input || !out_a || !out_b) return -1;
        memcpy([input contents], x, input_floats * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_iq2_pair_pipeline];
        uint32_t in32 = (uint32_t)in, rows32 = (uint32_t)rows;
        NSUInteger groups = (rows + 3) / 4;
        for (size_t i = 0; i < count; ++i) {
            if (experts[i] >= a->dims[2]) return -1;
            ds4f_tensor slice_a = *a;
            ds4f_tensor slice_b = *b;
            slice_a.file_offset += (uint64_t)experts[i] * expert_bytes_a;
            slice_b.file_offset += (uint64_t)experts[i] * expert_bytes_b;
            slice_a.nbytes = expert_bytes_a;
            slice_b.nbytes = expert_bytes_b;
            id<MTLBuffer> weight_a = get_weight(g, &slice_a);
            id<MTLBuffer> weight_b = get_weight(g, &slice_b);
            if (!weight_a || !weight_b) return -1;
            [enc setBuffer:weight_a offset:0 atIndex:0];
            [enc setBuffer:weight_b offset:0 atIndex:1];
            NSUInteger input_offset = (NSUInteger)(x_stride ? i * x_stride : 0) * sizeof(float);
            [enc setBuffer:input offset:input_offset atIndex:2];
            [enc setBuffer:g_iq2_signed_grid offset:0 atIndex:3];
            [enc setBuffer:out_a offset:i * rows * sizeof(float) atIndex:4];
            [enc setBuffer:out_b offset:i * rows * sizeof(float) atIndex:5];
            [enc setBytes:&in32 length:sizeof(in32) atIndex:6];
            [enc setBytes:&rows32 length:sizeof(rows32) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -1;
        const float *pa = (const float *)[out_a contents];
        const float *pb = (const float *)[out_b contents];
        for (size_t i = 0; i < count; ++i) {
            memcpy(ya + i * ya_stride, pa + i * rows, rows * sizeof(float));
            memcpy(yb + i * yb_stride, pb + i * rows, rows * sizeof(float));
        }
    }
    return 0;
}

int ds4f_metal_matvec_expert_q8k_batch(const ds4f_gguf *g,
                                       const ds4f_tensor *t,
                                       const uint32_t *experts, size_t count,
                                       const float *x, size_t x_stride,
                                       size_t in, float *y, size_t y_stride) {
    if (!t || getenv("DS4F_FORCE_CPU_EXPERTS")) return -1;
    if (metal_init() != 0) return -1;
    if (t->type == 16)
        return metal_expert_rows_batch(g, t, experts, count, x, x_stride, in,
                                       y, y_stride, g_iq2_pipeline, 1);
    if (t->type == 10)
        return metal_expert_rows_batch(g, t, experts, count, x, x_stride, in,
                                       y, y_stride, g_q2_pipeline, 0);
    return -1;
}

int ds4f_metal_matvec_expert_q8k_pair(const ds4f_gguf *g,
                                      const ds4f_tensor *a,
                                      const ds4f_tensor *b,
                                      const uint32_t *experts, size_t count,
                                      const float *x, size_t x_stride, size_t in,
                                      float *ya, size_t ya_stride,
                                      float *yb, size_t yb_stride) {
    if (!a || !b || getenv("DS4F_FORCE_CPU_EXPERTS")) return -1;
    if (metal_init() != 0) return -1;
    if (a->type == 16)
        return metal_iq2_rows_pair(g, a, b, experts, count, x, x_stride, in,
                                   ya, ya_stride, yb, yb_stride);
    if (a->type == 10)
        return metal_expert_rows_pair(g, a, b, experts, count, x, x_stride, in,
                                      ya, ya_stride, yb, yb_stride,
                                      g_q2_pipeline, 0);
    return -1;
}

int ds4f_metal_matvec_expert_q8k(const ds4f_gguf *g, const ds4f_tensor *t,
                                 uint32_t expert, const float *x, size_t in,
                                 float *y) {
    return ds4f_metal_matvec_expert_q8k_batch(g, t, &expert, 1, x, 0, in, y,
                                              (size_t)(t ? t->dims[1] : 0));
}

int ds4f_metal_iq2_probe(const ds4f_gguf *g, const ds4f_tensor *t,
                          uint32_t expert, const float *x, size_t in, float *out_value) {
    if (!g || !t || !x || !out_value || t->type != 16 || t->n_dims != 3 ||
        expert >= t->dims[2] || in != t->dims[0] || in % 256u ||
        t->nbytes % t->dims[2]) {
        fprintf(stderr, "ds4f: IQ2 probe rejected its arguments\n");
        return -1;
    }
    @autoreleasepool {
        if (metal_init() != 0 || !g_iq2_probe_pipeline) {
            fprintf(stderr, "ds4f: IQ2 probe Metal initialization failed\n");
            return -1;
        }
        const uint64_t expert_bytes = t->nbytes / t->dims[2];
        ds4f_tensor slice = *t;
        slice.file_offset += (uint64_t)expert * expert_bytes;
        slice.nbytes = expert_bytes;
        id<MTLBuffer> weight = get_weight(g, &slice);
        id<MTLBuffer> input = [g_device newBufferWithLength:in * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [g_device newBufferWithLength:sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (!weight || !input || !output) {
            fprintf(stderr, "ds4f: IQ2 probe buffer allocation failed\n");
            return -1;
        }
        memcpy([input contents], x, in * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_iq2_probe_pipeline];
        [enc setBuffer:weight offset:0 atIndex:0];
        [enc setBuffer:input offset:0 atIndex:1];
        [enc setBuffer:g_iq2_signed_grid offset:0 atIndex:2];
        [enc setBuffer:output offset:0 atIndex:3];
        const uint32_t in32 = (uint32_t)in;
        [enc setBytes:&in32 length:sizeof(in32) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "ds4f: IQ2 probe command failed: %s\n", cb.error.localizedDescription.UTF8String);
            return -1;
        }
        *out_value = *(const float *)[output contents];
    }
    return 0;
}
