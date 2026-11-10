#extension GL_EXT_control_flow_attributes : require
/* Compensate the rounding in 1+x, especially when softplus sees x << 1. */
float q36_log1p(float x) {
    precise float u = 1.0 + x;
    if (u == 1.0) return x;
    return log(u) * (x / (u - 1.0));
}

/* Round at the same F32 boundaries as the CPU graph. Native reciprocal,
 * exp and half packing can choose different approximations on RADV. */
float q36_div(float a, float b) { return float(double(a) / double(b)); }
float q36_sqrt(float v) { return float(sqrt(double(v))); }
float q36_rsqrt(float v) { return q36_div(1.0, q36_sqrt(v)); }
float q36_exp_series(float v) {
    if (v >= 89.0) return 1.0 / 0.0;
    if (v <= -104.0) return 0.0;
    int e = int(roundEven(double(v) * 1.4426950408889634074));
    double r = double(v) - double(e) * 0.69314718055994530942;
    double sum = 1.0, term = 1.0;
    [[unroll]] for (int k = 1; k <= 16; k++) { term *= r / double(k); sum += term; }
    return float(ldexp(sum, e));
}
/* Horner evaluation avoids sixteen FP64 divisions. Near an F32 midpoint,
 * retain the original series so quantization sees the same rounding. */
float q36_exp(float v) {
    if (v < -80.0 || v > 80.0) return q36_exp_series(v);
    int e=int(roundEven(double(v)*1.4426950408889634074));
    double r=double(v)-double(e)*0.69314718055994530942;
    double value=2.08767569878681e-09LF;
    value=fma(value,r,2.505210838544172e-08LF);
    value=fma(value,r,2.755731922398589e-07LF);
    value=fma(value,r,2.7557319223985893e-06LF);
    value=fma(value,r,2.48015873015873e-05LF);
    value=fma(value,r,0.0001984126984126984LF);
    value=fma(value,r,0.001388888888888889LF);
    value=fma(value,r,0.008333333333333333LF);
    value=fma(value,r,0.041666666666666664LF);
    value=fma(value,r,0.16666666666666666LF);
    value=fma(value,r,0.5LF);
    value=fma(value,r,1.0LF);
    value=fma(value,r,1.0LF);
    value=ldexp(value,e);
    float rounded=float(value);
    uint bits=floatBitsToUint(rounded);
    double adjacent=double(uintBitsToFloat(double(rounded)>value ? bits-1 : bits+1));
    double midpoint=(double(rounded)+adjacent)*0.5;
    if (abs(value-midpoint)<abs(value)*1e-12LF) return q36_exp_series(v);
    return rounded;
}
float q36_sigmoid(float v) { precise float den = 1.0 + q36_exp(-v); return q36_div(1.0,den); }
float q36_silu(float v) { precise float den = 1.0 + q36_exp(-v); return q36_div(v,den); }
uint q36_half(float v) {
    uint bits = floatBitsToUint(v), sign = (bits >> 16) & 32768u;
    uint mag = bits & 0x7fffffffu;
    if (mag >= 0x7f800000u) return sign | (mag == 0x7f800000u ? 31744u : 32256u);
    if (mag < 0x33000001u) return sign;
    int exponent = int(mag >> 23) - 127;
    uint mant = (mag & 0x7fffffu) | 0x800000u;
    uint shift = uint(max(13,-exponent-1));
    uint rounded = (mant + ((1u << (shift-1)) - 1u) + ((mant >> shift) & 1u)) >> shift;
    if (exponent < -14) return sign | rounded;
    uint half_value = uint(exponent+14) * 1024u + rounded;
    return sign | min(half_value,31744u);
}
