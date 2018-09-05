/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <cmath>
#include <functional>
#include <string>
#include "paddle/fluid/platform/cpu_info.h"
#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

namespace paddle {
namespace operators {
namespace math {

#ifdef __AVX__
// #include "paddle/legacy/cuda/src/avx_mathfun.h"
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))

#define _PI32AVX_CONST(Name, Val)                                          \
  static const ALIGN32_BEG int _pi32avx_##Name[4] ALIGN32_END = {Val, Val, \
                                                                 Val, Val}

_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS256_CONST(Name, Val)                                   \
  static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = { \
      Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST256(Name, Val)                                  \
  static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = { \
      Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS256_CONST_TYPE(Name, Type, Val)                       \
  static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = { \
      Val, Val, Val, Val, Val, Val, Val, Val}

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1);
_PS256_CONST(cephes_log_q1, -2.12194440e-4);
_PS256_CONST(cephes_log_q2, 0.693359375);

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

#ifndef __AVX2__
typedef union imm_xmm_union {
  __m256i imm;
  __m128i xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_)       \
  {                                               \
    imm_xmm_union u __attribute__((aligned(32))); \
    u.imm = imm_;                                 \
    xmm0_ = u.xmm[0];                             \
    xmm1_ = u.xmm[1];                             \
  }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_)       \
  {                                               \
    imm_xmm_union u __attribute__((aligned(32))); \
    u.xmm[0] = xmm0_;                             \
    u.xmm[1] = xmm1_;                             \
    imm_ = u.imm;                                 \
  }

#define AVX2_BITOP_USING_SSE2(fn)                           \
  static inline __m256i avx2_mm256_##fn(__m256i x, int a) { \
    /* use SSE2 instruction to perform the bitop AVX2 */    \
    __m128i x1, x2;                                         \
    __m256i ret;                                            \
    COPY_IMM_TO_XMM(x, x1, x2);                             \
    x1 = _mm_##fn(x1, a);                                   \
    x2 = _mm_##fn(x2, a);                                   \
    COPY_XMM_TO_IMM(x1, x2, ret);                           \
    return (ret);                                           \
  }

// #warning "Using SSE2 to perform AVX2 bitshift ops"
AVX2_BITOP_USING_SSE2(slli_epi32)
AVX2_BITOP_USING_SSE2(srli_epi32)

#define AVX2_INTOP_USING_SSE2(fn)                                     \
  static inline __m256i avx2_mm256_##fn(__m256i x, __m256i y) {       \
    /* use SSE2 instructions to perform the AVX2 integer operation */ \
    __m128i x1, x2;                                                   \
    __m128i y1, y2;                                                   \
    __m256i ret;                                                      \
    COPY_IMM_TO_XMM(x, x1, x2);                                       \
    COPY_IMM_TO_XMM(y, y1, y2);                                       \
    x1 = _mm_##fn(x1, y1);                                            \
    x2 = _mm_##fn(x2, y2);                                            \
    COPY_XMM_TO_IMM(x1, x2, ret);                                     \
    return (ret);                                                     \
  }

// #warning "Using SSE2 to perform AVX2 integer ops"
AVX2_INTOP_USING_SSE2(and_si128)
AVX2_INTOP_USING_SSE2(andnot_si128)
AVX2_INTOP_USING_SSE2(cmpeq_epi32)
AVX2_INTOP_USING_SSE2(sub_epi32)
AVX2_INTOP_USING_SSE2(add_epi32)
#define avx2_mm256_and_si256 avx2_mm256_and_si128
#define avx2_mm256_andnot_si256 avx2_mm256_andnot_si128
#else
#define avx2_mm256_slli_epi32 _mm256_slli_epi32
#define avx2_mm256_srli_epi32 _mm256_srli_epi32
#define avx2_mm256_and_si256 _mm256_and_si256
#define avx2_mm256_andnot_si256 _mm256_andnot_si256
#define avx2_mm256_cmpeq_epi32 _mm256_cmpeq_epi32
#define avx2_mm256_sub_epi32 _mm256_sub_epi32
#define avx2_mm256_add_epi32 _mm256_add_epi32
#endif /* __AVX2__ */

inline __m256 exp256_ps(__m256 x) {
  __m256 tmp = _mm256_setzero_ps(), fx;
  __m256i imm0;
  __m256 one = *reinterpret_cast<const __m256*>(_ps256_1);

  x = _mm256_min_ps(x, *reinterpret_cast<const __m256*>(_ps256_exp_hi));
  x = _mm256_max_ps(x, *reinterpret_cast<const __m256*>(_ps256_exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, *reinterpret_cast<const __m256*>(_ps256_cephes_LOG2EF));
  fx = _mm256_add_ps(fx, *reinterpret_cast<const __m256*>(_ps256_0p5));

  /* how to perform a floorf with SSE: just below */
  // imm0 = _mm256_cvttps_epi32(fx);
  // tmp  = _mm256_cvtepi32_ps(imm0);

  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  // __m256 mask = _mm256_cmpgt_ps(tmp, fx);
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp =
      _mm256_mul_ps(fx, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_C1));
  __m256 z =
      _mm256_mul_ps(fx, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_C2));
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x, x);

  __m256 y = *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p0);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p1));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p2));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p3));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p4));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p5));
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = avx2_mm256_add_epi32(
      imm0, *reinterpret_cast<const __m256i*>(_pi32_256_0x7f));
  imm0 = avx2_mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
#endif

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0

#define AVX_FLOAT_BLOCK 8
#define AVX_DOUBLE_BLOCK 4
#define AVX2_FLOAT_BLOCK 8
#define AVX2_DOUBLE_BLOCK 4
#define AVX512_FLOAT_BLOCK 16
#define AVX512_DOUBLE_BLOCK 8

template <typename T>
inline void vec_exp(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

template <typename T>
inline void vec_scal(const int n, const T a, T* x) {
  for (int i = 0; i < n; ++i) {
    x[i] = a * x[i];
  }
}

#ifdef PADDLE_WITH_MKLML
template <>
inline void vec_exp<float>(const int n, const float* x, float* y) {
  platform::dynload::vsExp(n, x, y);
}

template <>
inline void vec_exp<double>(const int n, const double* x, double* y) {
  platform::dynload::vdExp(n, x, y);
}

template <>
inline void vec_scal<float>(const int n, const float a, float* x) {
  platform::dynload::cblas_sscal(n, a, x, 1);
}

template <>
inline void vec_scal<double>(const int n, const double a, double* x) {
  platform::dynload::cblas_dscal(n, a, x, 1);
}
#endif

// MKL scal only support inplace, choose this if src and dst are not equal
template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_scal(const int n, const T a, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i];
  }
}

template <>
inline void vec_scal<float, platform::jit::avx>(const int n, const float a,
                                                const float* x, float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_scal<float, platform::jit::isa_any>(n, a, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 scalar = _mm256_set1_ps(a);
  __m256 tmp;
#define MOVE_ONE_STEP               \
  tmp = _mm256_loadu_ps(x + i);     \
  tmp = _mm256_mul_ps(tmp, scalar); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    y[i] = a * x[i];
  }
#else
  vec_scal<float, platform::jit::isa_any>(n, a, x, y);
#endif
}

template <>
inline void vec_scal<float, platform::jit::avx2>(const int n, const float a,
                                                 const float* x, float* y) {
  vec_scal<float, platform::jit::avx>(n, a, x, y);
}

template <>
inline void vec_scal<float, platform::jit::avx512_common>(const int n,
                                                          const float a,
                                                          const float* x,
                                                          float* y) {
  // TODO(TJ): enable me
  vec_scal<float, platform::jit::avx2>(n, a, x, y);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_bias_sub(const int n, const T a, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a - x[i];
  }
}

template <>
inline void vec_bias_sub<float, platform::jit::avx>(const int n, const float a,
                                                    const float* x, float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_bias_sub<float, platform::jit::isa_any>(n, a, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 bias = _mm256_set1_ps(a);
  __m256 tmp;
#define MOVE_ONE_STEP             \
  tmp = _mm256_loadu_ps(x + i);   \
  tmp = _mm256_sub_ps(bias, tmp); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    y[i] = a - x[i];
  }
#else
  vec_bias_sub<float, platform::jit::isa_any>(n, a, x, y);
#endif
}

template <>
inline void vec_bias_sub<float, platform::jit::avx2>(const int n, const float a,
                                                     const float* x, float* y) {
  vec_bias_sub<float, platform::jit::avx>(n, a, x, y);
}

template <>
inline void vec_bias_sub<float, platform::jit::avx512_common>(const int n,
                                                              const float a,
                                                              const float* x,
                                                              float* y) {
  // TODO(TJ): enable me
  vec_bias_sub<float, platform::jit::avx2>(n, a, x, y);
}

// out = x*y + (1-x)*z
template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_cross(const int n, const T* x, const T* y, const T* z, T* out) {
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * y[i] + (static_cast<T>(1) - x[i]) * z[i];
  }
}

template <>
inline void vec_cross<float, platform::jit::avx>(const int n, const float* x,
                                                 const float* y, const float* z,
                                                 float* out) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_cross<float, platform::jit::isa_any>(n, x, y, z, out);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 bias = _mm256_set1_ps(1.f);
  __m256 tmpx, tmpy, tmpz;
  for (i = 0; i < end; i += block) {
    tmpx = _mm256_loadu_ps(x + i);
    tmpy = _mm256_loadu_ps(y + i);
    tmpz = _mm256_loadu_ps(z + i);
    tmpy = _mm256_mul_ps(tmpx, tmpy);
    tmpx = _mm256_sub_ps(bias, tmpx);
    tmpz = _mm256_mul_ps(tmpx, tmpz);
    tmpz = _mm256_add_ps(tmpy, tmpz);
    _mm256_storeu_ps(out + i, tmpz);
  }
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    out[i] = x[i] * y[i] + (1.f - x[i]) * z[i];
  }
#else
  vec_cross<float, platform::jit::isa_any>(n, x, y, z, out);
#endif
}

template <>
inline void vec_cross<float, platform::jit::avx2>(const int n, const float* x,
                                                  const float* y,
                                                  const float* z, float* out) {
  vec_cross<float, platform::jit::avx>(n, x, y, z, out);
}

template <>
inline void vec_cross<float, platform::jit::avx512_common>(
    const int n, const float* x, const float* y, const float* z, float* out) {
  // TODO(TJ): enable me
  vec_cross<float, platform::jit::avx>(n, x, y, z, out);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_add_bias(const int n, const T a, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] + a;
  }
}

template <>
inline void vec_add_bias<float, platform::jit::avx>(const int n, const float a,
                                                    const float* x, float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_add_bias<float, platform::jit::isa_any>(n, a, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 bias = _mm256_set1_ps(a);
  __m256 tmp;
#define MOVE_ONE_STEP             \
  tmp = _mm256_loadu_ps(x + i);   \
  tmp = _mm256_add_ps(tmp, bias); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    y[i] = x[i] + a;
  }
#else
  vec_add_bias<float, platform::jit::isa_any>(n, a, x, y);
#endif
}

template <>
inline void vec_add_bias<float, platform::jit::avx2>(const int n, const float a,
                                                     const float* x, float* y) {
  vec_add_bias<float, platform::jit::avx>(n, a, x, y);
}

template <>
inline void vec_add_bias<float, platform::jit::avx512_common>(const int n,
                                                              const float a,
                                                              const float* x,
                                                              float* y) {
  // TODO(TJ): enable me
  vec_add_bias<float, platform::jit::avx2>(n, a, x, y);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_identity(const int n, const T* x, T* y) {
  // do nothing
  return;
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_sigmoid(const int n, const T* x, T* y) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  vec_exp<T>(n, y, y);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <>
inline void vec_sigmoid<float, platform::jit::avx>(const int n, const float* x,
                                                   float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_sigmoid<float, platform::jit::isa_any>(n, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
  __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
  __m256 zeros = _mm256_setzero_ps();
  __m256 tmp;
#define MOVE_ONE_STEP              \
  tmp = _mm256_loadu_ps(x + i);    \
  tmp = _mm256_max_ps(tmp, min);   \
  tmp = _mm256_min_ps(tmp, max);   \
  tmp = _mm256_sub_ps(zeros, tmp); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
  if (rest != 0) {
    i = n - block;
    MOVE_ONE_STEP;
  }

#undef MOVE_ONE_STEP
#define MOVE_ONE_STEP           \
  tmp = _mm256_loadu_ps(y + i); \
  tmp = exp256_ps(tmp);         \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
  if (rest != 0) {
    i = n - block;
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP

  __m256 ones = _mm256_set1_ps(1.0f);
#define MOVE_ONE_STEP             \
  tmp = _mm256_loadu_ps(y + i);   \
  tmp = _mm256_add_ps(ones, tmp); \
  tmp = _mm256_div_ps(ones, tmp); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step
  for (i = n - rest; i < n; ++i) {
    y[i] = 1.f / (1.f + y[i]);
  }
#else
  vec_sigmoid<float, platform::jit::isa_any>(n, x, y);
#endif
}

template <>
inline void vec_sigmoid<float, platform::jit::avx2>(const int n, const float* x,
                                                    float* y) {
  vec_sigmoid<float, platform::jit::avx>(n, x, y);
}

template <>
inline void vec_sigmoid<float, platform::jit::avx512_common>(const int n,
                                                             const float* x,
                                                             float* y) {
  // TODO(TJ): enable me
  vec_sigmoid<float, platform::jit::avx2>(n, x, y);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_tanh(const int n, const T* x, T* y) {
  vec_scal<T, isa>(n, static_cast<T>(2), x, y);
  vec_sigmoid<T, isa>(n, y, y);
  vec_scal<T>(n, static_cast<T>(2), y);
  vec_add_bias<T, isa>(n, static_cast<T>(-1), y, y);
}

// TODO(TJ): make relu clip
template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_relu(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

template <>
inline void vec_relu<float, platform::jit::avx>(const int n, const float* x,
                                                float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block * 4) {
    vec_relu<float, platform::jit::isa_any>(n, x, y);
    return;
  }

  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 zeros = _mm256_setzero_ps();
  __m256 tmp;
#define MOVE_ONE_STEP              \
  tmp = _mm256_loadu_ps(x + i);    \
  tmp = _mm256_max_ps(tmp, zeros); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
  if (rest == 0) {
    return;
  }
  i = n - block;
  MOVE_ONE_STEP;
#undef MOVE_ONE_STEP

#else
  vec_relu<float, platform::jit::isa_any>(n, x, y);
#endif
}

template <>
inline void vec_relu<float, platform::jit::avx2>(const int n, const float* x,
                                                 float* y) {
  vec_relu<float, platform::jit::avx>(n, x, y);
}

template <>
inline void vec_relu<float, platform::jit::avx512_common>(const int n,
                                                          const float* x,
                                                          float* y) {
  // TODO(TJ): enable me
  vec_relu<float, platform::jit::avx2>(n, x, y);
}

// TODO(TJ): optimize double of sigmoid, tanh and relu if necessary

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
class VecActivations {
 public:
  std::function<void(const int, const T*, T*)> operator()(
      const std::string& type) {
    if (type == "sigmoid") {
      return vec_sigmoid<T, isa>;
    } else if (type == "relu") {
      return vec_relu<T, isa>;
    } else if (type == "tanh") {
      return vec_tanh<T, isa>;
    } else if (type == "identity" || type == "") {
      return vec_identity<T, isa>;
    }
    LOG(FATAL) << "Not support type: " << type;
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
