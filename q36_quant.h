#ifndef Q36_QUANT_H
#define Q36_QUANT_H

#include <stdbool.h>
#include <stdint.h>

#ifndef Q36_QUANT_LINKAGE
#define Q36_QUANT_LINKAGE
#endif

Q36_QUANT_LINKAGE float q36_quant_f16_to_f32(uint16_t h);
Q36_QUANT_LINKAGE uint16_t q36_quant_f32_to_f16(float f);
Q36_QUANT_LINKAGE void q36_quant_q8_0(const float *x, void *out, int64_t n);
Q36_QUANT_LINKAGE void q36_quant_q4_0(const float *x, void *out, int64_t n);
Q36_QUANT_LINKAGE void q36_quant_q4_0_kv(const float *x, void *out, int64_t n);
Q36_QUANT_LINKAGE void q36_quant_q8_k(const float *x, void *out, int64_t n);
Q36_QUANT_LINKAGE float q36_quant_dot_q8_0(const void *a, const void *b, int n);
Q36_QUANT_LINKAGE bool q36_quant_dequantize(
        uint32_t type, const void *src, float *out, uint32_t n);
Q36_QUANT_LINKAGE void q36_quant_pack_iq1_grid(void *dst);

#endif
