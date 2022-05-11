#define __HIP_PLATFORM_AMD__

#include "hip/hip_runtime.h"
#include <stdint.h>
#include <iostream>
#include <assert.h>

void check_hip_error(void) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr
                << "Error: "
                << hipGetErrorString(err)
                << std::endl;
        exit(err);
    }
}

#define DEVICE __device__
#define GLOBAL
#define KERNEL extern "C" __global__
#define LOCAL __shared__
#define CONSTANT __constant__

#define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x
#define GET_GROUP_ID() blockIdx.x
#define GET_LOCAL_ID() threadIdx.x
#define GET_LOCAL_SIZE() blockDim.x
#define BARRIER_LOCAL() __syncthreads()

typedef unsigned char uchar;

#define HIP

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

__host__ __device__ uint32_t vcc = 0;
__host__ __device__ uint64_t check = 0x100000000ULL;

// Returns a + b, puts the carry in b
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
    ulong lo;
    uint32_t a_low = a;
    uint32_t b_low = *b;
    uint32_t add_low, add_high, temp;
    asm volatile("v_add_u32 %0, %1, %2;" : "+v"(add_low) : "v"(a_low), "v"(b_low));
    uint32_t a_high = a >> 32;
    uint32_t b_high = (*b) >> 32;
    asm volatile("v_addc_co_u32 %0, vcc, %1, %2, vcc;" : "+v"(add_high) : "v"(a_high), "v"(b_high): "vcc");
    asm volatile("v_addc_co_u32 %0, vcc, 0, 0, vcc;" : "+v"(temp)::"vcc");
    *b = temp;
    lo = add_high;
    lo = lo << 32;
    lo = lo + add_low;
    return lo;
}

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
    ulong lo;
    uint32_t a_lo, a_hi, b_lo, b_hi, c_lo, c_hi;
    a_lo = a;
    a_hi = a >> 32;
    b_lo = b;
    b_hi = b >> 32;
    c_lo = c;
    c_hi = c >> 32;

    uint32_t temp;
    uint32_t temp_carry;
    asm volatile("v_mul_lo_u32 %0, %1, %2;" : "+v"(temp): "v"(a_lo), "v"(b_lo));
    lo = temp;

    asm volatile("v_mul_lo_u32 %0, %1, %2;" : "+v"(temp): "v"(a_hi), "v"(b_lo));
    ulong res_temp1 = temp;
    asm volatile("v_mul_lo_u32 %0, %1, %2;" : "+v"(temp): "v"(a_lo), "v"(b_hi));
    ulong res_temp2 = temp;
    lo = add_with_carry_64(res_temp1, &res_temp2);
    lo = lo << 32;
    lo = add_with_carry_64(lo, &res_temp2);

    asm volatile("v_addc_co_u32 %0, vcc, 0, 0, vcc;" : "+v"(temp)::"vcc");
    *d = temp;
    return lo;
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
    ulong res = (ulong) a * b + c + *d;
    *d = res >> 32;
    return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
    uint lo, hi;
    asm volatile("v_add_u32 %0, %1, %2;" : "+v"(lo): "v"(a), "v"(*b));
    asm volatile("v_addc_co_u32 %0, vcc, 0, 0, vcc;" : "+v"(hi)::"vcc");
    *b = hi;
    return lo;
}

typedef uint uint32_t;
typedef int int32_t;
typedef uint limb;

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    uint64_t t;
    t = (uint64_t)a + (uint64_t)b;
    if (t >= check) {
        vcc = 1;
    } else {
	vcc = 0;
    }

    r = t;
    //asm volatile("v_add_co_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    return r;
}

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    uint64_t t;

    t = (uint64_t)a + (uint64_t)b + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
	vcc = 0;
    }

    r = t;
    //asm volatile("v_addc_co_u32 %0, vcc, %1, %2, vcc;" : "+v"(r): "v"(a), "v"(b): "vcc");
    return r;
}

DEVICE inline uint32_t addc(uint32_t a, uint32_t b) {
    uint32_t r;
    uint64_t t;

    t = (uint64_t)a + (uint64_t)b + (uint64_t)vcc;
    r = t;

    vcc = 0;

    //asm volatile("v_addc_co_u32 %0, vcc, %1, %2, vcc;" : "+v"(r): "v"(a), "v"(b): "vcc");
    ////To reset carry
    //asm volatile("v_add_co_u32 %0, %0, %1;" : "+v"(r) : "v"(0));
    return r;
}


DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
    //RR TODO:
    //Not used anywhere??????????
    uint32_t r;
    asm volatile("v_mul_lo_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(r): "v"(c));
    return r;
}

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    uint64_t t;
    r=a*b;
    t = (uint64_t)r + (uint64_t)c;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }

    r = t;
    //asm volatile("v_mul_lo_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    //asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(r): "v"(c));
    return r;
}

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    uint64_t t;
    r=a*b;

    t = (uint64_t)r + (uint64_t)c + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }

    r = t;
    //asm volatile("v_mul_lo_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(r): "v"(c): "vcc");
    return r;
}

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
    //RR TODO:
    //Not used anywhere??????????
    uint32_t r;
    asm volatile("v_mul_lo_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(r): "v"(c): "vcc");
    return r;
}

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
    //RR TODO:
    //Not used anywhere??????????
    uint32_t r;
    asm volatile("v_mul_hi_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    asm volatile("v_add_u32 %0, %0, %1" : "+v"(r): "v"(c));
    return r;
}

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    uint64_t t;
    t=(uint64_t)a*(uint64_t)b;
    r=t>>32;

    t = (uint64_t)r + (uint64_t)c;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }

    r = t;

    //asm volatile("v_mul_hi_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    //asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(r): "v"(c));
    return r;
}

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    uint64_t t;
    t=(uint64_t)a*(uint64_t)b;
    r=t>>32; 

    t = (uint64_t)r + (uint64_t)c + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }

    r = t;
    //asm volatile("v_mul_hi_u32 %0, %1, %2;" : "+v"(r): "v"(a), "v"(b));
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc;" : "+v"(r): "v"(c): "vcc");
    return r;
}

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    uint64_t t;
    t=(uint64_t)a*(uint64_t)b;
    r=t>>32; 

    t = (uint64_t)r + (uint64_t)c + (uint64_t)vcc;
    r = t;
    vcc = 0;
    //asm volatile("v_mul_hi_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(r): "v"(c): "vcc");
    //To reset carry
    //asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(r): "v"(0));
    return r;
}

typedef struct {
    int32_t _position;
} chain_t;

DEVICE inline
void chain_init(chain_t *c) {
    c->_position = 0;
}

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) {
    uint32_t r;

    ch->_position++;
    if (ch->_position == 1)
        r = add_cc(a, b);
    else
        r = addc_cc(a, b);
    return r;
}

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;

    ch->_position++;
    if (ch->_position == 1)
        r = madlo_cc(a, b, c);
    else
        r = madloc_cc(a, b, c);
    return r;
}

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;

    ch->_position++;
    if (ch->_position == 1)
        r = madhi_cc(a, b, c);
    else
        r = madhic_cc(a, b, c);
    return r;
}


#define Fr_limb uint
#define Fr_LIMBS 8
#define Fr_LIMB_BITS 32
#define Fr_INV 4294967295
typedef struct {
    Fr_limb val[Fr_LIMBS];
} Fr;
CONSTANT Fr Fr_ONE = {{4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881}};
CONSTANT Fr Fr_P = {{1, 4294967295, 4294859774, 1404937218, 161601541, 859428872, 698187080, 1944954707}};
CONSTANT Fr Fr_R2 = {{4092763245, 3382307216, 2274516003, 728559051, 1918122383, 97719446, 2673475345, 122214873}};
CONSTANT Fr Fr_ZERO = {{0, 0, 0, 0, 0, 0, 0, 0}};

DEVICE Fr Fr_sub_nvidia(Fr a, Fr b) {
    asm volatile("v_sub_co_u32 %0, %0, %1" : "+v"(a.val[0]): "v"(b.val[0]));
    asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[1]): "v"(b.val[1]): "vcc");
    asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[2]): "v"(b.val[2]): "vcc");
    asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[3]): "v"(b.val[3]): "vcc");
    asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[4]): "v"(b.val[4]): "vcc");
    asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[5]): "v"(b.val[5]): "vcc");
    asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[6]): "v"(b.val[6]): "vcc");
    asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[7]): "v"(b.val[7]): "vcc");
    asm volatile("v_sub_co_u32 %0, %0, %1" : "+v"(a.val[7]): "v"(0));
    return a;
}

DEVICE Fr Fr_add_nvidia(Fr a, Fr b) {
    asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(a.val[0]): "v"(b.val[0]));
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[1]): "v"(b.val[1]): "vcc");
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[2]): "v"(b.val[2]): "vcc");
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[3]): "v"(b.val[3]): "vcc");
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[4]): "v"(b.val[4]): "vcc");
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[5]): "v"(b.val[5]): "vcc");
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[6]): "v"(b.val[6]): "vcc");
    asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[7]): "v"(b.val[7]): "vcc");
    asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(a.val[7]): "v"(0));
    return a;
}

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define Fr_BITS (Fr_LIMBS * Fr_LIMB_BITS)
#if Fr_LIMB_BITS == 32
#define Fr_mac_with_carry mac_with_carry_32
#define Fr_add_with_carry add_with_carry_32
#elif Fr_LIMB_BITS == 64
#define Fr_mac_with_carry mac_with_carry_64
#define Fr_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool Fr_gte(Fr a, Fr b) {
    for (char i = Fr_LIMBS - 1; i >= 0; i--) {
        //RR TODO: Recheck logic
        if (a.val[i] > b.val[i])
            return true;
        if (a.val[i] < b.val[i])
            return false;
    }
    return true;
}

// Equals
DEVICE bool Fr_eq(Fr a, Fr b) {
    for (uchar i = 0; i < Fr_LIMBS; i++)
        if (a.val[i] != b.val[i])
            return false;
    return true;
}

// Normal addition
#define Fr_add_ Fr_add_nvidia
#define Fr_sub_ Fr_sub_nvidia

// Modular subtraction

DEVICE Fr Fr_sub(Fr a, Fr b) {
    Fr res = Fr_sub_(a, b);
    if (!Fr_gte(a, b)) res = Fr_add_(res, Fr_P);
    return res;
}

// Modular addition
DEVICE Fr Fr_add(Fr a, Fr b) {
    Fr res = Fr_add_(a, b);
    if (Fr_gte(res, Fr_P)) res = Fr_sub_(res, Fr_P);
    return res;
}


#if defined(CUDA) || defined(HIP)
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void Fr_reduce(uint32_t accLow[Fr_LIMBS], uint32_t np0, uint32_t fq[Fr_LIMBS]) {
    // accLow is an IN and OUT vector
    // count must be even
    const uint32_t count = Fr_LIMBS;
    uint32_t accHigh[Fr_LIMBS];
    uint32_t bucket = 0, lowCarry = 0, highCarry = 0, q;
    int32_t i, j;

#pragma unroll
    for (i = 0; i < count; i++)
        accHigh[i] = 0;

    // bucket is used so we don't have to push a carry all the way down the line

#pragma unroll
    for (j = 0; j < count; j++) {       // main iteration
        if (j % 2 == 0) {
            add_cc(bucket, 0xFFFFFFFF);
            accLow[0] = addc_cc(accLow[0], accHigh[1]);
            bucket = addc(0, 0);

            q = accLow[0] * np0;

            chain_t chain1;
            chain_init(&chain1);

#pragma unroll
            for (i = 0; i < count; i += 2) {
                accLow[i] = chain_madlo(&chain1, q, fq[i], accLow[i]);
                accLow[i + 1] = chain_madhi(&chain1, q, fq[i], accLow[i + 1]);
            }
            lowCarry = chain_add(&chain1, 0, 0);

            chain_t chain2;
            chain_init(&chain2);
            for (i = 0; i < count - 2; i += 2) {
                accHigh[i] = chain_madlo(&chain2, q, fq[i + 1], accHigh[i + 2]);    // note the shift down
                accHigh[i + 1] = chain_madhi(&chain2, q, fq[i + 1], accHigh[i + 3]);
            }
            accHigh[i] = chain_madlo(&chain2, q, fq[i + 1], highCarry);
            accHigh[i + 1] = chain_madhi(&chain2, q, fq[i + 1], 0);
        } else {
            add_cc(bucket, 0xFFFFFFFF);
            accHigh[0] = addc_cc(accHigh[0], accLow[1]);
            bucket = addc(0, 0);

            q = accHigh[0] * np0;

            chain_t chain3;
            chain_init(&chain3);
#pragma unroll
            for (i = 0; i < count; i += 2) {
                accHigh[i] = chain_madlo(&chain3, q, fq[i], accHigh[i]);
                accHigh[i + 1] = chain_madhi(&chain3, q, fq[i], accHigh[i + 1]);
            }
            highCarry = chain_add(&chain3, 0, 0);

            chain_t chain4;
            chain_init(&chain4);
            for (i = 0; i < count - 2; i += 2) {
                accLow[i] = chain_madlo(&chain4, q, fq[i + 1], accLow[i + 2]);    // note the shift down
                accLow[i + 1] = chain_madhi(&chain4, q, fq[i + 1], accLow[i + 3]);
            }
            accLow[i] = chain_madlo(&chain4, q, fq[i + 1], lowCarry);
            accLow[i + 1] = chain_madhi(&chain4, q, fq[i + 1], 0);
        }
    }

    // at this point, accHigh needs to be shifted back a word and added to accLow
    // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
    // can just push it into the carry chain.

    chain_t chain5;
    chain_init(&chain5);
    chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
#pragma unroll
    for (i = 0; i < count - 1; i++)
        accLow[i] = chain_add(&chain5, accLow[i], accHigh[i + 1]);
    accLow[i] = chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void Fr_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
    const uint32_t xLimbs = Fr_LIMBS;
    const uint32_t yLimbs = Fr_LIMBS;
    const uint32_t xyLimbs = Fr_LIMBS * 2;
    uint32_t temp[Fr_LIMBS * 2];
    uint32_t carry = 0;

#pragma unroll
    for (int32_t i = 0; i < xyLimbs; i++) {
        temp[i] = 0;
    }

#pragma unroll
    for (int32_t i = 0; i < xLimbs; i++) {
        chain_t chain1;
        chain_init(&chain1);
#pragma unroll
        for (int32_t j = 0; j < yLimbs; j++) {
            if ((i + j) % 2 == 1) {
                temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
                temp[i + j] = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
            }
        }
        if (i % 2 == 1) {
            temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
        }
    }

#pragma unroll
    for (int32_t i = xyLimbs - 1; i > 0; i--) {
        temp[i] = temp[i - 1];
    }
    temp[0] = 0;

#pragma unroll
    for (int32_t i = 0; i < xLimbs; i++) {
        chain_t chain2;
        chain_init(&chain2);

#pragma unroll
        for (int32_t j = 0; j < yLimbs; j++) {
            if ((i + j) % 2 == 0) {
                temp[i + j] = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
                temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
            }
        }
        if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
            temp[i + yLimbs] = chain_add(&chain2, temp[i + yLimbs], carry);
            temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
            carry = chain_add(&chain2, 0, 0);
        }
        if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
            carry = chain_add(&chain2, carry, 0);
        }
    }

#pragma unroll
    for (int32_t i = 0; i < xyLimbs; i++) {
        xy[i] = temp[i];
    }
}

DEVICE Fr Fr_mul_nvidia(Fr a, Fr b) {
    // Perform full multiply
    limb ab[2 * Fr_LIMBS];
    Fr_mult_v1(a.val, b.val, ab);

    uint32_t io[Fr_LIMBS];
#pragma unroll
    for (int i = 0; i < Fr_LIMBS; i++) {
        io[i] = ab[i];
    }
    Fr_reduce(io, Fr_INV, Fr_P.val);

    // Add io to the upper words of ab
    ab[Fr_LIMBS] = add_cc(ab[Fr_LIMBS], io[0]);
    int j;
#pragma unroll
    for (j = 1; j < Fr_LIMBS - 1; j++) {
        ab[j + Fr_LIMBS] = addc_cc(ab[j + Fr_LIMBS], io[j]);
    }
    ab[2 * Fr_LIMBS - 1] = addc(ab[2 * Fr_LIMBS - 1], io[Fr_LIMBS - 1]);

    Fr r;
#pragma unroll
    for (int i = 0; i < Fr_LIMBS; i++) {
        r.val[i] = ab[i + Fr_LIMBS];
    }

    if (Fr_gte(r, Fr_P)) {
        r = Fr_sub_(r, Fr_P);
    }

    return r;
}

#endif

// Modular multiplication
DEVICE Fr Fr_mul_default(Fr a, Fr b) {
    /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
     * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
     * Learn more:
     * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
     * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
     */
    Fr_limb t[Fr_LIMBS + 2] = {0};
    for (uchar i = 0; i < Fr_LIMBS; i++) {
        Fr_limb carry = 0;
        for (uchar j = 0; j < Fr_LIMBS; j++)
            t[j] = Fr_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
        t[Fr_LIMBS] = Fr_add_with_carry(t[Fr_LIMBS], &carry);
        t[Fr_LIMBS + 1] = carry;

        carry = 0;
        Fr_limb m = Fr_INV * t[0];
        Fr_mac_with_carry(m, Fr_P.val[0], t[0], &carry);
        for (uchar j = 1; j < Fr_LIMBS; j++)
            t[j - 1] = Fr_mac_with_carry(m, Fr_P.val[j], t[j], &carry);

        t[Fr_LIMBS - 1] = Fr_add_with_carry(t[Fr_LIMBS], &carry);
        t[Fr_LIMBS] = t[Fr_LIMBS + 1] + carry;
    }

    Fr result;
    for (uchar i = 0; i < Fr_LIMBS; i++) result.val[i] = t[i];

    if (Fr_gte(result, Fr_P)) result = Fr_sub_(result, Fr_P);

    return result;
}

#if defined(CUDA) || defined(HIP)

DEVICE Fr Fr_mul(Fr a, Fr b) {
    return Fr_mul_nvidia(a, b);
}

#else
DEVICE Fr Fr_mul(Fr a, Fr b) {
  return Fr_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE Fr Fr_sqr(Fr a) {
    return Fr_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of Fr_add(a, a)
DEVICE Fr Fr_double(Fr a) {
    for (uchar i = Fr_LIMBS - 1; i >= 1; i--)
        a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (Fr_LIMB_BITS - 1));
    a.val[0] <<= 1;
    if (Fr_gte(a, Fr_P)) a = Fr_sub_(a, Fr_P);
    return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE Fr Fr_pow(Fr base, uint exponent) {
    Fr res = Fr_ONE;
    while (exponent > 0) {
        if (exponent & 1)
            res = Fr_mul(res, base);
        exponent = exponent >> 1;
        base = Fr_sqr(base);
    }
    return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE Fr Fr_pow_lookup(GLOBAL Fr *bases, uint exponent) {
    Fr res = Fr_ONE;
    uint i = 0;
    while (exponent > 0) {
        if (exponent & 1)
            res = Fr_mul(res, bases[i]);
        exponent = exponent >> 1;
        i++;
    }
    return res;
}

DEVICE Fr Fr_mont(Fr a) {
    return Fr_mul(a, Fr_R2);
}

DEVICE Fr Fr_unmont(Fr a) {
    Fr one = Fr_ZERO;
    one.val[0] = 1;
    return Fr_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool Fr_get_bit(Fr l, uint i) {
    return (l.val[Fr_LIMBS - 1 - i / Fr_LIMB_BITS] >> (Fr_LIMB_BITS - 1 - (i % Fr_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint Fr_get_bits(Fr l, uint skip, uint window) {
    uint ret = 0;
    for (uint i = 0; i < window; i++) {
        ret <<= 1;
        ret |= Fr_get_bit(l, skip + i);
    }
    return ret;
}


#define Fq_limb uint
#define Fq_LIMBS 12
#define Fq_LIMB_BITS 32
#define Fq_INV 4294770685
typedef struct {
    Fq_limb val[Fq_LIMBS];
} Fq;
CONSTANT Fq Fq_ONE = {
        {196605, 1980301312, 3289120770, 3958636555, 1405573306, 1598593111, 1884444485, 2010011731, 2723605613,
         1543969431, 4202751123, 368467651}};
CONSTANT Fq Fq_P = {
        {4294945451, 3120496639, 2975072255, 514588670, 4138792484, 1731252896, 4085584575, 1685539716, 1129032919,
         1260103606, 964683418, 436277738}};
CONSTANT Fq Fq_R2 = {
        {473175878, 4108263220, 164693233, 175564454, 1284880085, 2380613484, 2476573632, 1743489193, 3038352685,
         2591637125, 2462770090, 295210981}};
CONSTANT Fq Fq_ZERO = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};


DEVICE Fq Fq_sub_nvidia(Fq a, Fq b) {
    uint64_t t;
    //t = (uint64_t)a.val[0] - (uint64_t)b.val[0];
    if (b.val[0] > a.val[0]) {
        vcc = 1;
    } else {
        vcc = 0;
    }

    a.val[0] = a.val[0] - b.val[0];

    t = (uint64_t)b.val[1] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[1]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[1] = a.val[1] - t;

    t = (uint64_t)b.val[2] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[2]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[2] = a.val[2] - t;

    t = (uint64_t)b.val[3] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[3]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[3] = a.val[3] - t;

    t = (uint64_t)b.val[4] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[4]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[4] = a.val[4] - t;

    t = (uint64_t)b.val[5] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[5]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[5] = a.val[5] - t;

    t = (uint64_t)b.val[6] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[6]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[6] = a.val[6] - t;

    t = (uint64_t)b.val[7] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[7]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[7] = a.val[7] - t;

    t = (uint64_t)b.val[8] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[8]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[8] = a.val[8] - t;

    t = (uint64_t)b.val[9] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[9]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[9] = a.val[9] - t;

    t = (uint64_t)b.val[10] + (uint64_t)vcc;
    if (t > (uint64_t)a.val[10]) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[10] = a.val[10] - t;

    t = (uint64_t)b.val[11] + (uint64_t)vcc;
    a.val[11] = a.val[11] - t;

    vcc = 0;

    //asm volatile("v_sub_co_u32 %0, %0, %1" : "+v"(a.val[0]): "v"(b.val[0]));
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[1]): "v"(b.val[1]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[2]): "v"(b.val[2]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[3]): "v"(b.val[3]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[4]): "v"(b.val[4]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[5]): "v"(b.val[5]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[6]): "v"(b.val[6]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[7]): "v"(b.val[7]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[8]): "v"(b.val[8]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[9]): "v"(b.val[9]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[10]): "v"(b.val[10]): "vcc");
    //asm volatile("v_subb_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[11]): "v"(b.val[11]): "vcc");
    //asm volatile("v_sub_co_u32 %0, %0, %1" : "+v"(a.val[11]): "v"(0));
    return a;
}

DEVICE Fq Fq_add_nvidia(Fq a, Fq b) {
    uint64_t t;
    t = (uint64_t)a.val[0] + (uint64_t)b.val[0];
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[0] = t;

    t = (uint64_t)a.val[1] + (uint64_t)b.val[1] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[1] = t;

    t = (uint64_t)a.val[2] + (uint64_t)b.val[2] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[2] = t;

    t = (uint64_t)a.val[3] + (uint64_t)b.val[3] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[3] = t;

    t = (uint64_t)a.val[4] + (uint64_t)b.val[4] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[4] = t;

    t = (uint64_t)a.val[5] + (uint64_t)b.val[5] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[5] = t;

    t = (uint64_t)a.val[6] + (uint64_t)b.val[6] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[6] = t;

    t = (uint64_t)a.val[7] + (uint64_t)b.val[7] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[7] = t;

    t = (uint64_t)a.val[8] + (uint64_t)b.val[8] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[8] = t;

    t = (uint64_t)a.val[9] + (uint64_t)b.val[9] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[9] = t;

    t = (uint64_t)a.val[10] + (uint64_t)b.val[10] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[10] = t;

    t = (uint64_t)a.val[11] + (uint64_t)b.val[11] + (uint64_t)vcc;
    if (t >= check) {
        vcc = 1;
    } else {
        vcc = 0;
    }
    a.val[11] = t;

    vcc = 0;

    //asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(a.val[0]): "v"(b.val[0]));
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[1]): "v"(b.val[1]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[2]): "v"(b.val[2]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[3]): "v"(b.val[3]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[4]): "v"(b.val[4]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[5]): "v"(b.val[5]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[6]): "v"(b.val[6]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[7]): "v"(b.val[7]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[8]): "v"(b.val[8]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[9]): "v"(b.val[9]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[10]): "v"(b.val[10]): "vcc");
    //asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(a.val[11]): "v"(b.val[11]): "vcc");
    //asm volatile("v_add_co_u32 %0, %0, %1" : "+v"(a.val[11]): "v"(0));
    return a;
}

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define Fq_BITS (Fq_LIMBS * Fq_LIMB_BITS)
#if Fq_LIMB_BITS == 32
#define Fq_mac_with_carry mac_with_carry_32
#define Fq_add_with_carry add_with_carry_32
#elif Fq_LIMB_BITS == 64
#define Fq_mac_with_carry mac_with_carry_64
#define Fq_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool Fq_gte(Fq a, Fq b) {
    for (char i = Fq_LIMBS - 1; i >= 0; i--) {
        //RR TODO: Recheck logic
        if (a.val[i] > b.val[i])
            return true;
        if (a.val[i] < b.val[i])
            return false;
    }
    return true;
}

// Equals
DEVICE bool Fq_eq(Fq a, Fq b) {
    for (uchar i = 0; i < Fq_LIMBS; i++)
        if (a.val[i] != b.val[i])
            return false;
    return true;
}

// Normal addition
#define Fq_add_ Fq_add_nvidia
#define Fq_sub_ Fq_sub_nvidia

// Modular subtraction

DEVICE Fq Fq_sub(Fq a, Fq b) {
    Fq res = Fq_sub_(a, b);
    if (!Fq_gte(a, b)) res = Fq_add_(res, Fq_P);
    return res;
}

// Modular addition
DEVICE Fq Fq_add(Fq a, Fq b) {
    Fq res = Fq_add_(a, b);
    if (Fq_gte(res, Fq_P)) res = Fq_sub_(res, Fq_P);
    return res;
}


#if defined(CUDA) || defined(HIP)
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void Fq_reduce(uint32_t accLow[Fq_LIMBS], uint32_t np0, uint32_t fq[Fq_LIMBS]) {
    // accLow is an IN and OUT vector
    // count must be even
    const uint32_t count = Fq_LIMBS;
    uint32_t accHigh[Fq_LIMBS];
    uint32_t bucket = 0, lowCarry = 0, highCarry = 0, q;
    int32_t i, j;

#pragma unroll
    for (i = 0; i < count; i++)
        accHigh[i] = 0;

    // bucket is used so we don't have to push a carry all the way down the line

#pragma unroll
    for (j = 0; j < count; j++) {       // main iteration
        if (j % 2 == 0) {
            add_cc(bucket, 0xFFFFFFFF);
            accLow[0] = addc_cc(accLow[0], accHigh[1]);
            bucket = addc(0, 0);

            q = accLow[0] * np0;

            chain_t chain1;
            chain_init(&chain1);

#pragma unroll
            for (i = 0; i < count; i += 2) {
                accLow[i] = chain_madlo(&chain1, q, fq[i], accLow[i]);
                accLow[i + 1] = chain_madhi(&chain1, q, fq[i], accLow[i + 1]);
            }
            lowCarry = chain_add(&chain1, 0, 0);

            chain_t chain2;
            chain_init(&chain2);
            for (i = 0; i < count - 2; i += 2) {
                accHigh[i] = chain_madlo(&chain2, q, fq[i + 1], accHigh[i + 2]);    // note the shift down
                accHigh[i + 1] = chain_madhi(&chain2, q, fq[i + 1], accHigh[i + 3]);
            }
            accHigh[i] = chain_madlo(&chain2, q, fq[i + 1], highCarry);
            accHigh[i + 1] = chain_madhi(&chain2, q, fq[i + 1], 0);
        } else {
            add_cc(bucket, 0xFFFFFFFF);
            accHigh[0] = addc_cc(accHigh[0], accLow[1]);
            bucket = addc(0, 0);

            q = accHigh[0] * np0;

            chain_t chain3;
            chain_init(&chain3);
#pragma unroll
            for (i = 0; i < count; i += 2) {
                accHigh[i] = chain_madlo(&chain3, q, fq[i], accHigh[i]);
                accHigh[i + 1] = chain_madhi(&chain3, q, fq[i], accHigh[i + 1]);
            }
            highCarry = chain_add(&chain3, 0, 0);

            chain_t chain4;
            chain_init(&chain4);
            for (i = 0; i < count - 2; i += 2) {
                accLow[i] = chain_madlo(&chain4, q, fq[i + 1], accLow[i + 2]);    // note the shift down
                accLow[i + 1] = chain_madhi(&chain4, q, fq[i + 1], accLow[i + 3]);
            }
            accLow[i] = chain_madlo(&chain4, q, fq[i + 1], lowCarry);
            accLow[i + 1] = chain_madhi(&chain4, q, fq[i + 1], 0);
        }
    }

    // at this point, accHigh needs to be shifted back a word and added to accLow
    // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
    // can just push it into the carry chain.

    chain_t chain5;
    chain_init(&chain5);
    chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
#pragma unroll
    for (i = 0; i < count - 1; i++)
        accLow[i] = chain_add(&chain5, accLow[i], accHigh[i + 1]);
    accLow[i] = chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void Fq_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
    const uint32_t xLimbs = Fq_LIMBS;
    const uint32_t yLimbs = Fq_LIMBS;
    const uint32_t xyLimbs = Fq_LIMBS * 2;
    uint32_t temp[Fq_LIMBS * 2];
    uint32_t carry = 0;

#pragma unroll
    for (int32_t i = 0; i < xyLimbs; i++) {
        temp[i] = 0;
    }

#pragma unroll
    for (int32_t i = 0; i < xLimbs; i++) {
        chain_t chain1;
        chain_init(&chain1);
#pragma unroll
        for (int32_t j = 0; j < yLimbs; j++) {
            if ((i + j) % 2 == 1) {
                temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
                temp[i + j] = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
            }
        }
        if (i % 2 == 1) {
            temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
        }
    }

#pragma unroll
    for (int32_t i = xyLimbs - 1; i > 0; i--) {
        temp[i] = temp[i - 1];
    }
    temp[0] = 0;

#pragma unroll
    for (int32_t i = 0; i < xLimbs; i++) {
        chain_t chain2;
        chain_init(&chain2);

#pragma unroll
        for (int32_t j = 0; j < yLimbs; j++) {
            if ((i + j) % 2 == 0) {
                temp[i + j] = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
                temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
            }
        }
        if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
            temp[i + yLimbs] = chain_add(&chain2, temp[i + yLimbs], carry);
            temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
            carry = chain_add(&chain2, 0, 0);
        }
        if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
            carry = chain_add(&chain2, carry, 0);
        }
    }

#pragma unroll
    for (int32_t i = 0; i < xyLimbs; i++) {
        xy[i] = temp[i];
    }
}

DEVICE Fq Fq_mul_nvidia(Fq a, Fq b) {
    // Perform full multiply
    limb ab[2 * Fq_LIMBS];
    Fq_mult_v1(a.val, b.val, ab);

    uint32_t io[Fq_LIMBS];
#pragma unroll
    for (int i = 0; i < Fq_LIMBS; i++) {
        io[i] = ab[i];
    }
    Fq_reduce(io, Fq_INV, Fq_P.val);

    // Add io to the upper words of ab
    ab[Fq_LIMBS] = add_cc(ab[Fq_LIMBS], io[0]);
    int j;
#pragma unroll
    for (j = 1; j < Fq_LIMBS - 1; j++) {
        ab[j + Fq_LIMBS] = addc_cc(ab[j + Fq_LIMBS], io[j]);
    }
    ab[2 * Fq_LIMBS - 1] = addc(ab[2 * Fq_LIMBS - 1], io[Fq_LIMBS - 1]);

    Fq r;
#pragma unroll
    for (int i = 0; i < Fq_LIMBS; i++) {
        r.val[i] = ab[i + Fq_LIMBS];
    }

    if (Fq_gte(r, Fq_P)) {
        r = Fq_sub_(r, Fq_P);
    }

    return r;
}

#endif

// Modular multiplication
DEVICE Fq Fq_mul_default(Fq a, Fq b) {
    /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
     * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
     * Learn more:
     * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
     * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
     */
    Fq_limb t[Fq_LIMBS + 2] = {0};
    for (uchar i = 0; i < Fq_LIMBS; i++) {
        Fq_limb carry = 0;
        for (uchar j = 0; j < Fq_LIMBS; j++)
            t[j] = Fq_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
        t[Fq_LIMBS] = Fq_add_with_carry(t[Fq_LIMBS], &carry);
        t[Fq_LIMBS + 1] = carry;

        carry = 0;
        Fq_limb m = Fq_INV * t[0];
        Fq_mac_with_carry(m, Fq_P.val[0], t[0], &carry);
        for (uchar j = 1; j < Fq_LIMBS; j++)
            t[j - 1] = Fq_mac_with_carry(m, Fq_P.val[j], t[j], &carry);

        t[Fq_LIMBS - 1] = Fq_add_with_carry(t[Fq_LIMBS], &carry);
        t[Fq_LIMBS] = t[Fq_LIMBS + 1] + carry;
    }

    Fq result;
    for (uchar i = 0; i < Fq_LIMBS; i++) result.val[i] = t[i];

    if (Fq_gte(result, Fq_P)) result = Fq_sub_(result, Fq_P);

    return result;
}

DEVICE Fq Fq_mul(Fq a, Fq b) {
    return Fq_mul_nvidia(a, b);
}


// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE Fq Fq_sqr(Fq a) {
    return Fq_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of Fq_add(a, a)
DEVICE Fq Fq_double(Fq a) {
    for (uchar i = Fq_LIMBS - 1; i >= 1; i--)
        a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (Fq_LIMB_BITS - 1));
    a.val[0] <<= 1;
    if (Fq_gte(a, Fq_P)) a = Fq_sub_(a, Fq_P);
    return a;
}


// Elliptic curve operations (Short Weierstrass Jacobian form)

#define G1_ZERO ((G1_projective){Fq_ZERO, Fq_ONE, Fq_ZERO})

typedef struct {
    Fq x;
    Fq y;
} G1_affine;

typedef struct {
    Fq x;
    Fq y;
    Fq z;
} G1_projective;

DEVICE G1_projective G1_double(G1_projective inp) {
    const Fq local_zero = Fq_ZERO;
    if (Fq_eq(inp.z, local_zero)) {
        return inp;
    }

    const Fq a = Fq_sqr(inp.x); // A = X1^2
    const Fq b = Fq_sqr(inp.y); // B = Y1^2
    Fq c = Fq_sqr(b); // C = B^2

    // D = 2*((X1+B)2-A-C)
    Fq d = Fq_add(inp.x, b);
    d = Fq_sqr(d);
    d = Fq_sub(Fq_sub(d, a), c);
    d = Fq_double(d);

    const Fq e = Fq_add(Fq_double(a), a); // E = 3*A
    const Fq f = Fq_sqr(e);

    inp.z = Fq_mul(inp.y, inp.z);
    inp.z = Fq_double(inp.z); // Z3 = 2*Y1*Z1
    inp.x = Fq_sub(Fq_sub(f, d), d); // X3 = F-2*D

    // Y3 = E*(D-X3)-8*C
    c = Fq_double(c);
    c = Fq_double(c);
    c = Fq_double(c);
    inp.y = Fq_sub(Fq_mul(Fq_sub(d, inp.x), e), c);

    return inp;
}

DEVICE G1_projective G1_add_mixed(G1_projective a, G1_affine b) {
    const Fq local_zero = Fq_ZERO;
    if (Fq_eq(a.z, local_zero)) {
        const Fq local_one = Fq_ONE;
        a.x = b.x;
        a.y = b.y;
        a.z = local_one;
        return a;
    }

    const Fq z1z1 = Fq_sqr(a.z);
    const Fq u2 = Fq_mul(b.x, z1z1);
    const Fq s2 = Fq_mul(Fq_mul(b.y, a.z), z1z1);

    if (Fq_eq(a.x, u2) && Fq_eq(a.y, s2)) {
        return G1_double(a);
    }

    const Fq h = Fq_sub(u2, a.x); // H = U2-X1
    const Fq hh = Fq_sqr(h); // HH = H^2
    Fq i = Fq_double(hh);
    i = Fq_double(i); // I = 4*HH
    Fq j = Fq_mul(h, i); // J = H*I
    Fq r = Fq_sub(s2, a.y);
    r = Fq_double(r); // r = 2*(S2-Y1)
    const Fq v = Fq_mul(a.x, i);

    G1_projective ret;

    // X3 = r^2 - J - 2*V
    ret.x = Fq_sub(Fq_sub(Fq_sqr(r), j), Fq_double(v));

    // Y3 = r*(V-X3)-2*Y1*J
    j = Fq_mul(a.y, j);
    j = Fq_double(j);
    ret.y = Fq_sub(Fq_mul(Fq_sub(v, ret.x), r), j);

    // Z3 = (Z1+H)^2-Z1Z1-HH
    ret.z = Fq_add(a.z, h);
    ret.z = Fq_sub(Fq_sub(Fq_sqr(ret.z), z1z1), hh);
    return ret;
}

__device__ void print_Fq(const char *name, Fq in, const char *end) {
    printf("%s=>", name);
    for (int i = 0; i < Fq_LIMBS; i++) {
        printf("%u,", in.val[i]);
    }
    printf("%s", end);
}

__device__ void print_G1(const char *name, G1_projective g1, const char *end) {
    printf("%s==>", name);
    print_Fq("x", g1.x, "*");
    print_Fq("y", g1.y, "*");
    print_Fq("z", g1.z, "\n");
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
//[[clang::optnone]]
DEVICE G1_projective G1_add(G1_projective a, G1_projective b) {
    // print_G1("a",a,"\n");
    // print_G1("b",b,"\n");
    const Fq local_zero = Fq_ZERO;
    if (Fq_eq(a.z, local_zero)) return b;
    if (Fq_eq(b.z, local_zero)) return a;

    const Fq z1z1 = Fq_sqr(a.z); // Z1Z1 = Z1^2
    const Fq z2z2 = Fq_sqr(b.z); // Z2Z2 = Z2^2
    const Fq u1 = Fq_mul(a.x, z2z2); // U1 = X1*Z2Z2
    const Fq u2 = Fq_mul(b.x, z1z1); // U2 = X2*Z1Z1
    Fq s1 = Fq_mul(Fq_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
    const Fq s2 = Fq_mul(Fq_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

    if (Fq_eq(u1, u2) && Fq_eq(s1, s2))
        return G1_double(a);
    else {
        const Fq h = Fq_sub(u2, u1); // H = U2-U1
        Fq i = Fq_double(h);
        i = Fq_sqr(i); // I = (2*H)^2
        const Fq j = Fq_mul(h, i); // J = H*I
        Fq r = Fq_sub(s2, s1);
        r = Fq_double(r); // r = 2*(S2-S1)
        const Fq v = Fq_mul(u1, i); // V = U1*I
        a.x = Fq_sub(Fq_sub(Fq_sub(Fq_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

        // Y3 = r*(V - X3) - 2*S1*J
        a.y = Fq_mul(Fq_sub(v, a.x), r);
        s1 = Fq_mul(s1, j);
        s1 = Fq_double(s1); // S1 = S1 * J * 2
        a.y = Fq_sub(a.y, s1);

        // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
        a.z = Fq_add(a.z, b.z);
        a.z = Fq_sqr(a.z);
        a.z = Fq_sub(Fq_sub(a.z, z1z1), z2z2);
        a.z = Fq_mul(a.z, h);

        return a;
    }
}

KERNEL void kernel_G1_add_mixed(G1_projective *a, G1_affine *b, G1_projective *result) {
    *result = G1_add_mixed(*a, *b);
}

KERNEL void kernel_G1_add(G1_projective *a, G1_projective *b, G1_projective *result) {
    *result = G1_add(*a, *b);
}

void print_G1_project(G1_projective in) {
    std::cout << "x: ";
    for (int i = 0; i < 12; i++) {
        std::cout << in.x.val[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "y: ";
    for (int i = 0; i < 12; i++) {
        std::cout << in.y.val[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "z: ";
    for (int i = 0; i < 12; i++) {
        std::cout << in.z.val[i] << ", ";
    }
    std::cout << std::endl << std::endl;
}

void normal_print(uint *in) {
    for (int i = 0; i < 36; i++) {
        std::cout << in[i] << ", ";
    }
}

int main() {
    uint32_t a[36] = {0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 196605, 1980301312, 3289120770, 3958636555, 1405573306,
                      1598593111, 1884444485, 2010011731, 2723605613, 1543969431, 4202751123, 368467651, 0, 0, 0, 0, 0,
                      30, 0, 0, 0, 0, 0, 0};

    uint32_t b[36] = {0, 0, 0, 200, 0, 0, 0, 0, 0, 0, 0, 0, 196605, 1980301312, 3289120770, 3958636555, 1405573306,
                      1598593111, 1884444485, 2010011731, 2723605613, 1543969431, 4202751123, 368467651, 0, 0, 0, 0, 0,
                      40, 0, 0, 0, 0, 0, 0};

    uint32_t cudaExpected[36] = {695163763, 838428337, 867136025, 3916970060, 1083605276, 2882035772, 3603006931,
                                 2269309842, 422274527, 1169772790, 1990394245, 416975321, 1229022948, 3366429108,
                                 670218974, 1658335027, 392632874, 1379067484, 798160530, 3656524164, 3793686573,
                                 2144155088, 2721370348, 298035558, 413203031, 3318893592, 1282426328, 1145762026,
                                 1542369093, 485346739, 1679000480, 4026228341, 2371190916, 3558967189, 3094593878,
                                 414846589};


    G1_projective *a_d, *b_d, *result_d;
    G1_projective result;

    auto size = sizeof(G1_projective);
    hipMalloc(&a_d, size);
    hipMalloc(&b_d, size);
    hipMalloc(&result_d, size);
    check_hip_error();

    hipMemcpy(a_d, a, size, hipMemcpyHostToDevice);
    hipMemcpy(b_d, b, size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    check_hip_error();

    hipLaunchKernelGGL(kernel_G1_add, dim3(1), dim3(1), 0, 0, a_d, b_d, result_d);

    check_hip_error();
    hipDeviceSynchronize();
    check_hip_error();

    hipMemcpy(&result, result_d, size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    check_hip_error();
    print_G1_project(result);

    //flattening result
    uint32_t flattenResult[36];
    memcpy(flattenResult,&result,size);

    //assert cudaExpected vs result
    for (int i = 0; i < 36; ++i) {
        assert(flattenResult[i]==cudaExpected[i]);
    }

    printf("No error\n\n");
    hipFree(a_d);
    hipFree(b_d);
    hipFree(result_d);
}
