/* expf reduction and polynomial from Arm optimized-routines (MIT).
 * Copyright (c) 2017-2026, Arm Limited.
 * Match the host libm rounding before expert-weight normalization. */
float q36_router_exp(float x) {
    if (x < -104.0) return 0;
    if (x > 89.0) return 1.0/0.0;
    const double powers[32]=double[32](
        1.0LF,
        1.0218971486541166LF,
        1.0442737824274138LF,
        1.0671404006768237LF,
        1.0905077326652577LF,
        1.1143867425958924LF,
        1.1387886347566916LF,
        1.1637248587775775LF,
        1.189207115002721LF,
        1.215247359980469LF,
        1.241857812073484LF,
        1.2690509571917332LF,
        1.2968395546510096LF,
        1.3252366431597413LF,
        1.3542555469368927LF,
        1.383909881963832LF,
        1.4142135623730951LF,
        1.4451808069770467LF,
        1.4768261459394993LF,
        1.5091644275934228LF,
        1.5422108254079407LF,
        1.5759808451078865LF,
        1.6104903319492543LF,
        1.645755478153965LF,
        1.681792830507429LF,
        1.718619298122478LF,
        1.7562521603732995LF,
        1.7947090750031072LF,
        1.8340080864093424LF,
        1.8741676341103LF,
        1.9152065613971474LF,
        1.9571441241754002LF);
    precise double z=double(x)*46.16624130844683LF;
    double kd=roundEven(z);
    int k=int(kd);
    precise double r=z-kd;
    double scale=ldexp(powers[k&31],k>>5);
    precise double a=fma(1.6938359250920212e-06LF,r,0.00023459809789509004LF);
    precise double b=fma(0.021660849396613134LF,r,1.0LF);
    precise double value=fma(a,r*r,b)*scale;
    return float(value);
}

/* The first eight softmax lanes use the CPU vector exponential. */
/* Keep halfway conversions at the same F32 boundaries as the host. */
float q36_router_round(double value) {
    uvec2 v=unpackDouble2x32(value);
    int exponent=int((v.y>>20)&2047u)-896;
    if (exponent<=0 || exponent>=255) return float(value);
    uint bits=(v.y&0x80000000u)|(uint(exponent)<<23)|((v.y&0xfffffu)<<3)|(v.x>>29);
    uint tail=v.x&0x1fffffffu;
    bits+=uint(tail>0x10000000u || (tail==0x10000000u && (bits&1u)!=0));
    return uintBitsToFloat(bits);
}
float q36_router_fma(float a, float b, float c) {
    return q36_router_round(fma(double(a),double(b),double(c)));
}
float q36_router_vector_exp(float x) {
    if (!(x>=-80.0 && x<=80.0)) return q36_router_exp(x);
    precise float z=fma(x,1.4426950216293335,12582912.0);
    precise float n=z-12582912.0;
    precise float r=q36_router_fma(-n,1.428606765330187e-6,q36_router_fma(-n,0.693145751953125,x));
    float scale=uintBitsToFloat((floatBitsToUint(z)<<23)+0x3f800000u);
    precise float rr=r*r;
    precise float a=q36_router_fma(0.0082473903894424438,r,0.041899766772985458);
    precise float b=q36_router_fma(0.16668395698070526,r,0.49999126791954041);
    precise float c=q36_router_fma(a,rr,b);
    precise float tail=0.99999940395355225*r;
    precise float p=q36_router_fma(c,rr,tail);
    return q36_router_fma(p,scale,scale);
}
