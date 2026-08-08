#ifndef DS4F_QUANT_H
#define DS4F_QUANT_H

#include "ds4f_gguf.h"
#include <stddef.h>

float ds4f_f16_to_f32(uint16_t h);
const uint64_t *ds4f_iq2_grid_data(void);
int ds4f_tensor_load(const ds4f_gguf *g, const ds4f_tensor *t, void **data_out);
int ds4f_matvec(const ds4f_gguf *g, const ds4f_tensor *t, const float *x, float *y);
int ds4f_matvec_q8_pair(const ds4f_gguf *g, const ds4f_tensor *a,
                         const ds4f_tensor *b, const float *x,
                         float *ya, float *yb);
int ds4f_matvec_pair_shared_input(const ds4f_gguf *g, const ds4f_tensor *a,
                                  const ds4f_tensor *b, const float *x,
                                  float *ya, float *yb);
int ds4f_matvec_q8_0_cpu(const ds4f_gguf *g, const ds4f_tensor *t,
                          const float *x, float *y);
int ds4f_matvec_expert(const ds4f_gguf *g, const ds4f_tensor *t,
                       uint32_t expert, const float *x, float *y);
int ds4f_matvec_expert_q8k(const ds4f_gguf *g, const ds4f_tensor *t,
                           uint32_t expert, const float *x, size_t n, float *y);
int ds4f_matvec_expert_q8k_cpu(const ds4f_gguf *g, const ds4f_tensor *t,
                               uint32_t expert, const float *x, size_t n, float *y);
int ds4f_metal_iq2_probe(const ds4f_gguf *g, const ds4f_tensor *t,
                          uint32_t expert, const float *x, size_t n, float *out);
int ds4f_matvec_expert_q8k_batch(const ds4f_gguf *g, const ds4f_tensor *t,
                                 const uint32_t *experts, size_t count,
                                 const float *x, size_t x_stride, size_t n,
                                 float *y, size_t y_stride);
int ds4f_matvec_expert_q8k_pair(const ds4f_gguf *g,
                                const ds4f_tensor *a, const ds4f_tensor *b,
                                const uint32_t *experts, size_t count,
                                const float *x, size_t x_stride, size_t n,
                                float *ya, size_t ya_stride,
                                float *yb, size_t yb_stride);
int ds4f_matvec_expert_q8k_pair_prefetch(const ds4f_gguf *g,
                                         const ds4f_tensor *a, const ds4f_tensor *b,
                                         const uint32_t *experts, size_t count,
                                         const float *x, size_t x_stride, size_t n,
                                         float *ya, size_t ya_stride,
                                         float *yb, size_t yb_stride,
                                         const ds4f_tensor *prefetch);
int ds4f_matvec_group(const ds4f_gguf *g, const ds4f_tensor *t,
                      const float *x, uint32_t groups, uint32_t group_in,
                      uint32_t group_out, float *y);
void ds4f_rms_norm(float *out, const float *x, const float *w,
                   size_t n, float eps);
void ds4f_swiglu(float *out, const float *gate, const float *up,
                 size_t n, float clamp);

#endif
