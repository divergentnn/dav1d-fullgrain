/*
 * Film grain synthesis variants (bitdepth template). See src/fg_variant.h.
 *
 * NON-CONFORMANT: AV1 film grain synthesis is normative; these modes produce
 * different (statistically equivalent) grain than any conformant decoder.
 *
 * Every mode has a plain C path and, on x86-64 with AVX2, the same arithmetic
 * with AVX2 kernels (selected at runtime from dav1d's CPU flags, so
 * --cpumask / dav1d_set_cpu_flags_mask also apply). Both paths are
 * bit-identical; py/validate_variants.py checks them against a numpy model.
 */

#include "config.h"

#include <stdint.h>
#include <string.h>

#include "common/attributes.h"
#include "common/bitdepth.h"
#include "common/intops.h"

#include "src/fg_apply.h"
#include "src/fg_variant.h"
#include "src/filmgrain.h"
#include "src/tables.h"

#if ARCH_X86_64
#include <immintrin.h>
#include <x86intrin.h>
#define FGV_AVX2 __attribute__((target("avx2")))
#define FGV_TSC() __rdtsc()
#else
#if ARCH_AARCH64
#include <arm_neon.h>
#endif
#define FGV_TSC() 0
#endif

#define FGV_XL 16  /* full mode: warm-up columns left of the frame */
#define FGV_XR 8   /* full mode: extra columns right of the frame */

/* per-phase cycle counters (DAV1D_GRAIN_STATS=2) */
#define PH(i) do { if (fgv->stats > 1) { const uint64_t _t = FGV_TSC(); \
    atomic_fetch_add(&fgv->phase[i], _t - _tp); _tp = _t; } } while (0)

static inline int fgv_round2(const int x, const int shift) {
    return (x + ((1 << shift) >> 1)) >> shift;
}

static inline entry *tmpl_plane(const Dav1dFGVariant *const fgv, const int k,
                                const int pl)
{
    return (entry *) ((uint8_t *) fgv->tmpl + (size_t) k * fgv->tmpl_set_bytes) +
           pl * (GRAIN_HEIGHT + 1) * GRAIN_WIDTH;
}

/* ------------------------------------------------------------------------ */
/* per-frame preparation                                                      */

void bitfn(dav1d_fgv_prep)(const Dav1dFilmGrainDSPContext *const dsp,
                           Dav1dFGVariant *const fgv, const Dav1dPicture *const in,
                           const Dav1dFilmGrainData *const data)
{
#if BITDEPTH != 8
    const int bitdepth_max = (1 << in->p.bpc) - 1;
#endif
    if (fgv->mode != DAV1D_FGMODE_MULTI && fgv->mode != DAV1D_FGMODE_DUAL) return;
    /* K templates from derived seeds, through the normal (asm) generators,
     * so every template has exactly the signalled AR statistics */
    Dav1dFilmGrainData d = *data;
    for (int k = 0; k < fgv->ntmpl; k++) {
        d.seed = dav1d_fgv_tmpl_seed(data->seed, k);
        entry (*const ty)[GRAIN_WIDTH] = (entry (*)[GRAIN_WIDTH]) tmpl_plane(fgv, k, 0);
        dsp->generate_grain_y(ty, &d HIGHBD_TAIL_SUFFIX);
        for (int uv = 0; uv < 2; uv++)
            if (in->p.layout != DAV1D_PIXEL_LAYOUT_I400 &&
                (data->num_uv_points[uv] || data->chroma_scaling_from_luma))
                dsp->generate_grain_uv[in->p.layout - 1](
                    (entry (*)[GRAIN_WIDTH]) tmpl_plane(fgv, k, 1 + uv),
                    (const entry (*)[GRAIN_WIDTH]) ty, &d, uv HIGHBD_TAIL_SUFFIX);
    }
}

/* scaling LUTs premultiplied for pmulhrsw (sc << (15 - scaling_shift)) and
 * the Gaussian table pre-shifted for the full-frame mode */
void bitfn(dav1d_fgv_prep_luts)(Dav1dFGVariant *const fgv, const Dav1dPicture *const in,
                                const Dav1dFilmGrainData *const data,
                                const uint8_t scaling[3][SCALING_SIZE])
{
    if (fgv->mode < DAV1D_FGMODE_MULTI) return;
#if BITDEPTH != 8
    const int bitdepth_max = (1 << in->p.bpc) - 1;
#endif
    const int bitdepth_min_8 = bitdepth_from_max(bitdepth_max) - 8;
    if (fgv->simd && fgv->lut32) {
        const int n = BITDEPTH_MAX + 1, sh = 15 - data->scaling_shift;
        for (int pl = 0; pl < 3; pl++)
            for (int i = 0; i < n; i++)
                fgv->lut32[pl * 4096 + i] = scaling[pl][i] << sh;
    }
    if (fgv->mode == DAV1D_FGMODE_FULL) {
        const int gshift = 4 - bitdepth_min_8 + data->grain_scale_shift;
        for (int i = 0; i < 2048; i++)
            fgv->gtab[i] = fgv_round2(dav1d_gaussian_sequence[i], gshift);
    }
}

/* ------------------------------------------------------------------------ */
/* scaling + clipping (identical to dav1d's fgy / fguv add_noise)             */

typedef struct FGApply {
    const uint8_t *scaling;
    const int32_t *lut32;     /* NULL: C path */
    int shift, min_value, max_value;
    /* chroma */
    int sx, csfl, uv_luma_mult, uv_mult, uv_offset, pmax;
} FGApply;

#if ARCH_X86_64
static inline FGV_AVX2 __m256i lut_pack16(const int32_t *const lut,
                                          const __m256i i0, const __m256i i1)
{
    const __m256i l0 = _mm256_i32gather_epi32((const int *) lut, i0, 4);
    const __m256i l1 = _mm256_i32gather_epi32((const int *) lut, i1, 4);
    return _mm256_permute4x64_epi64(_mm256_packs_epi32(l0, l1), 0xD8);
}

static inline FGV_AVX2 int apply_y_avx2(pixel *const dst, const pixel *const src,
                                        const int16_t *const g, const int w,
                                        const FGApply *const A)
{
    const __m256i vmn = _mm256_set1_epi16(A->min_value);
    const __m256i vmx = _mm256_set1_epi16(A->max_value);
    int x = 0;
    for (; x + 16 <= w; x += 16) {
        __m256i s, i0, i1;
#if BITDEPTH == 8
        const __m128i s8 = _mm_loadu_si128((const __m128i *) (src + x));
        s = _mm256_cvtepu8_epi16(s8);
        i0 = _mm256_cvtepu8_epi32(s8);
        i1 = _mm256_cvtepu8_epi32(_mm_srli_si128(s8, 8));
#else
        s = _mm256_loadu_si256((const __m256i *) (src + x));
        i0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s));
        i1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s, 1));
#endif
        const __m256i l = lut_pack16(A->lut32, i0, i1);
        const __m256i n = _mm256_mulhrs_epi16(l, _mm256_loadu_si256((const __m256i *) (g + x)));
        __m256i d = _mm256_add_epi16(s, n);
        d = _mm256_min_epi16(_mm256_max_epi16(d, vmn), vmx);
#if BITDEPTH == 8
        _mm_storeu_si128((__m128i *) (dst + x),
                         _mm_packus_epi16(_mm256_castsi256_si128(d),
                                          _mm256_extracti128_si256(d, 1)));
#else
        _mm256_storeu_si256((__m256i *) (dst + x), d);
#endif
    }
    return x;
}

static inline FGV_AVX2 int apply_uv_avx2(pixel *const dst, const pixel *const src,
                                         const pixel *const luma,
                                         const int16_t *const g, const int w,
                                         const FGApply *const A)
{
    const __m256i vmn = _mm256_set1_epi16(A->min_value);
    const __m256i vmx = _mm256_set1_epi16(A->max_value);
    const __m256i ones = _mm256_set1_epi16(1), one = _mm256_set1_epi32(1);
    const __m256i lm = _mm256_set1_epi32(A->uv_luma_mult), m = _mm256_set1_epi32(A->uv_mult);
    const __m256i off = _mm256_set1_epi32(A->uv_offset), pmax = _mm256_set1_epi32(A->pmax);
    const __m256i zero = _mm256_setzero_si256();
    int x = 0;
    for (; x + 16 <= w; x += 16) {
        __m256i a0, a1, s, s0, s1;
        if (A->sx) {
#if BITDEPTH == 8
            const __m256i Lb = _mm256_loadu_si256((const __m256i *) (luma + 2 * x));
            const __m256i L0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(Lb));
            const __m256i L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(Lb, 1));
#else
            const __m256i L0 = _mm256_loadu_si256((const __m256i *) (luma + 2 * x));
            const __m256i L1 = _mm256_loadu_si256((const __m256i *) (luma + 2 * x + 16));
#endif
            a0 = _mm256_srli_epi32(_mm256_add_epi32(_mm256_madd_epi16(L0, ones), one), 1);
            a1 = _mm256_srli_epi32(_mm256_add_epi32(_mm256_madd_epi16(L1, ones), one), 1);
        } else {
#if BITDEPTH == 8
            const __m128i L = _mm_loadu_si128((const __m128i *) (luma + x));
            a0 = _mm256_cvtepu8_epi32(L);
            a1 = _mm256_cvtepu8_epi32(_mm_srli_si128(L, 8));
#else
            const __m256i L = _mm256_loadu_si256((const __m256i *) (luma + x));
            a0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(L));
            a1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(L, 1));
#endif
        }
#if BITDEPTH == 8
        const __m128i s8 = _mm_loadu_si128((const __m128i *) (src + x));
        s = _mm256_cvtepu8_epi16(s8);
        s0 = _mm256_cvtepu8_epi32(s8);
        s1 = _mm256_cvtepu8_epi32(_mm_srli_si128(s8, 8));
#else
        s = _mm256_loadu_si256((const __m256i *) (src + x));
        s0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s));
        s1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s, 1));
#endif
        if (!A->csfl) {
            a0 = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(a0, lm),
                                                                     _mm256_mullo_epi32(s0, m)), 6), off);
            a1 = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(a1, lm),
                                                                     _mm256_mullo_epi32(s1, m)), 6), off);
            a0 = _mm256_min_epi32(_mm256_max_epi32(a0, zero), pmax);
            a1 = _mm256_min_epi32(_mm256_max_epi32(a1, zero), pmax);
        }
        const __m256i l = lut_pack16(A->lut32, a0, a1);
        const __m256i n = _mm256_mulhrs_epi16(l, _mm256_loadu_si256((const __m256i *) (g + x)));
        __m256i d = _mm256_add_epi16(s, n);
        d = _mm256_min_epi16(_mm256_max_epi16(d, vmn), vmx);
#if BITDEPTH == 8
        _mm_storeu_si128((__m128i *) (dst + x),
                         _mm_packus_epi16(_mm256_castsi256_si128(d),
                                          _mm256_extracti128_si256(d, 1)));
#else
        _mm256_storeu_si256((__m256i *) (dst + x), d);
#endif
    }
    return x;
}
#endif

#if ARCH_AARCH64
/* NEON: vqrdmulh(sc << (15 - shift), g) == round2(sc * g, shift), exactly
 * like pmulhrsw on x86. NEON has no gather, so the scaling LUT is read with
 * scalar loads. */
static inline int apply_y_neon(pixel *const dst, const pixel *const src,
                               const int16_t *const g, const int w,
                               const FGApply *const A)
{
    const int16x8_t vmn = vdupq_n_s16(A->min_value), vmx = vdupq_n_s16(A->max_value);
    const int16x8_t sh = vdupq_n_s16(15 - A->shift);
    const uint8_t *const sc = A->scaling;
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        uint8_t l8[8];
        for (int i = 0; i < 8; i++) l8[i] = sc[src[x + i]];
#if BITDEPTH == 8
        const int16x8_t s = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + x)));
#else
        const int16x8_t s = vreinterpretq_s16_u16(vld1q_u16(src + x));
#endif
        const int16x8_t l = vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(l8))), sh);
        const int16x8_t n = vqrdmulhq_s16(l, vld1q_s16(g + x));
        const int16x8_t d = vminq_s16(vmaxq_s16(vqaddq_s16(s, n), vmn), vmx);
#if BITDEPTH == 8
        vst1_u8(dst + x, vqmovun_s16(d));
#else
        vst1q_u16(dst + x, vreinterpretq_u16_s16(d));
#endif
    }
    return x;
}

static inline int apply_uv_neon(pixel *const dst, const pixel *const src,
                                const pixel *const luma, const int16_t *const g,
                                const int w, const FGApply *const A)
{
    const int16x8_t vmn = vdupq_n_s16(A->min_value), vmx = vdupq_n_s16(A->max_value);
    const int16x8_t sh = vdupq_n_s16(15 - A->shift);
    const int32x4_t off = vdupq_n_s32(A->uv_offset), zero = vdupq_n_s32(0);
    const int32x4_t pmax = vdupq_n_s32(A->pmax);
    const uint8_t *const sc = A->scaling;
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        uint16x8_t avg;
        if (A->sx) {
#if BITDEPTH == 8
            avg = vrshrq_n_u16(vpaddlq_u8(vld1q_u8(luma + 2 * x)), 1);
#else
            avg = vrshrq_n_u16(vpaddq_u16(vld1q_u16(luma + 2 * x), vld1q_u16(luma + 2 * x + 8)), 1);
#endif
        } else {
#if BITDEPTH == 8
            avg = vmovl_u8(vld1_u8(luma + x));
#else
            avg = vld1q_u16(luma + x);
#endif
        }
#if BITDEPTH == 8
        const int16x8_t s = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + x)));
#else
        const int16x8_t s = vreinterpretq_s16_u16(vld1q_u16(src + x));
#endif
        uint16_t val[8];
        if (!A->csfl) {
            int32x4_t c0 = vmulq_n_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(avg))), A->uv_luma_mult);
            int32x4_t c1 = vmulq_n_s32(vreinterpretq_s32_u32(vmovl_high_u16(avg)), A->uv_luma_mult);
            c0 = vmlaq_n_s32(c0, vmovl_s16(vget_low_s16(s)), A->uv_mult);
            c1 = vmlaq_n_s32(c1, vmovl_high_s16(s), A->uv_mult);
            c0 = vminq_s32(vmaxq_s32(vaddq_s32(vshrq_n_s32(c0, 6), off), zero), pmax);
            c1 = vminq_s32(vmaxq_s32(vaddq_s32(vshrq_n_s32(c1, 6), off), zero), pmax);
            vst1q_u16(val, vcombine_u16(vqmovun_s32(c0), vqmovun_s32(c1)));
        } else {
            vst1q_u16(val, avg);
        }
        uint8_t l8[8];
        for (int i = 0; i < 8; i++) l8[i] = sc[val[i]];
        const int16x8_t l = vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(l8))), sh);
        const int16x8_t n = vqrdmulhq_s16(l, vld1q_s16(g + x));
        const int16x8_t d = vminq_s16(vmaxq_s16(vqaddq_s16(s, n), vmn), vmx);
#if BITDEPTH == 8
        vst1_u8(dst + x, vqmovun_s16(d));
#else
        vst1q_u16(dst + x, vreinterpretq_u16_s16(d));
#endif
    }
    return x;
}
#endif

static ALWAYS_INLINE void apply_y_rows(pixel *const dst_row, const pixel *const src_row,
                                       const ptrdiff_t stride, const int16_t *const g0,
                                       const ptrdiff_t gstride, const int w, const int h,
                                       const FGApply *const A, const int simd)
{
    for (int y = 0; y < h; y++) {
        const pixel *const src = src_row + y * PXSTRIDE(stride);
        pixel *const dst = dst_row + y * PXSTRIDE(stride);
        const int16_t *const g = g0 + y * gstride;
        int x = 0;
#if ARCH_X86_64
        if (simd) x = apply_y_avx2(dst, src, g, w, A);
#elif ARCH_AARCH64
        if (simd) x = apply_y_neon(dst, src, g, w, A);
#endif
        for (; x < w; x++) {
            const int noise = fgv_round2(A->scaling[src[x]] * g[x], A->shift);
            dst[x] = iclip(src[x] + noise, A->min_value, A->max_value);
        }
    }
}

static ALWAYS_INLINE void apply_uv_rows(pixel *const dst_row, const pixel *const src_row,
                                        const ptrdiff_t stride,
                                        const pixel *const luma_row,
                                        const ptrdiff_t luma_stride, const int sy,
                                        const int16_t *const g0, const ptrdiff_t gstride,
                                        const int w, const int h,
                                        const FGApply *const A, const int simd
                                        HIGHBD_DECL_SUFFIX)
{
    const int sx = A->sx;
    for (int y = 0; y < h; y++) {
        const pixel *const src = src_row + y * PXSTRIDE(stride);
        pixel *const dst = dst_row + y * PXSTRIDE(stride);
        const pixel *const luma = luma_row + (y << sy) * PXSTRIDE(luma_stride);
        const int16_t *const g = g0 + y * gstride;
        int x = 0;
#if ARCH_X86_64
        if (simd) x = apply_uv_avx2(dst, src, luma, g, w, A);
#elif ARCH_AARCH64
        if (simd) x = apply_uv_neon(dst, src, luma, g, w, A);
#endif
        for (; x < w; x++) {
            int avg = luma[x << sx];
            if (sx) avg = (avg + luma[(x << sx) + 1] + 1) >> 1;
            int val = avg;
            if (!A->csfl) {
                const int combined = avg * A->uv_luma_mult + src[x] * A->uv_mult;
                val = iclip_pixel((combined >> 6) + A->uv_offset);
            }
            const int noise = fgv_round2(A->scaling[val] * g[x], A->shift);
            dst[x] = iclip(src[x] + noise, A->min_value, A->max_value);
        }
    }
}

static void setup_apply(FGApply *const A, const Dav1dFGVariant *const fgv,
                        const Dav1dFilmGrainData *const data,
                        const uint8_t *const scaling, const int lut_pl,
                        const int is_uv, const int uv, const int is_id, const int sx
                        HIGHBD_DECL_SUFFIX)
{
    const int bitdepth_min_8 = bitdepth_from_max(bitdepth_max) - 8;
    A->scaling = scaling;
    A->lut32 = fgv->simd && fgv->lut32 ? fgv->lut32 + lut_pl * 4096 : NULL;
    A->shift = data->scaling_shift;
    if (data->clip_to_restricted_range) {
        A->min_value = 16 << bitdepth_min_8;
        A->max_value = (is_uv ? (is_id ? 235 : 240) : 235) << bitdepth_min_8;
    } else {
        A->min_value = 0;
        A->max_value = BITDEPTH_MAX;
    }
    A->sx = sx;
    A->csfl = data->chroma_scaling_from_luma;
    A->uv_luma_mult = is_uv ? data->uv_luma_mult[uv] : 0;
    A->uv_mult = is_uv ? data->uv_mult[uv] : 0;
    A->uv_offset = is_uv ? data->uv_offset[uv] * (1 << bitdepth_min_8) : 0;
    A->pmax = BITDEPTH_MAX;
}

/* ------------------------------------------------------------------------ */
/* block layers (multi / dual)                                                */

typedef struct FGLayer {
    const Dav1dFGVariant *fgv;
    int pl;
    unsigned seed;
    int layer, Kl, kbase;
    int goff_y, goff_x;       /* grid offset of the layer, plane pixels */
    int bsy, bsx, ovy, ovx;   /* block size and overlap width */
    int Sy, Sx;               /* extended patch size (block + overlap) */
    int subx, suby;
    int overlap;
    const int (*wx)[2], (*wy)[2];
    int gmin, gmax;
} FGLayer;

typedef struct FGBlk {
    const entry *t;  /* template at the patch's top-left (oy, ox) */
    int sign, rot;
} FGBlk;

static const int fgv_w_full[2][2] = { { 27, 17 }, { 17, 27 } };
static const int fgv_w_sub[2][2]  = { { 23, 22 }, { 0, 0 } };

static ALWAYS_INLINE void blk_get(FGBlk *const b, const FGLayer *const L,
                                  const int by, const int bx)
{
    const uint32_t h = dav1d_fgv_blk_hash(L->seed, L->layer, by, bx);
    const int rv = h & 0xFF;
    const int k = L->kbase + (int) ((h >> 8) % (unsigned) L->Kl);
    const int ox = 3 + (2 >> L->subx) * (3 + (rv >> 4));
    const int oy = 3 + (2 >> L->suby) * (3 + (rv & 15));
    b->t = tmpl_plane(L->fgv, k, L->pl) + oy * GRAIN_WIDTH + ox;
    b->sign = (h >> 16) & 1;
    b->rot = (h >> 17) & 1;
}

static ALWAYS_INLINE int blk_val(const FGBlk *const b, const FGLayer *const L,
                                 const int ly, const int lx)
{
    const int yy = b->rot ? L->Sy - 1 - ly : ly;
    const int xx = b->rot ? L->Sx - 1 - lx : lx;
    const int v = b->t[yy * GRAIN_WIDTH + xx];
    return b->sign ? -v : v;
}

static ALWAYS_INLINE int blend2(const int old, const int cur, const int *const w,
                                const int gmin, const int gmax)
{
    return iclip(fgv_round2(old * w[0] + cur * w[1], 5), gmin, gmax);
}

/* Fill a w x h window at plane position (Y0, X0) with the layer's unscaled
 * grain (dav1d overlap blend at the layer's own block seams). */
static ALWAYS_INLINE void layer_fill(int16_t *const buf, const ptrdiff_t bstr,
                                     const int Y0, const int X0, const int h, const int w,
                                     const FGLayer *const L)
{
    const int by0 = (Y0 + L->goff_y) / L->bsy;
    const int by1 = (Y0 + h - 1 + L->goff_y) / L->bsy;
    const int bx0 = (X0 + L->goff_x) / L->bsx;
    const int bx1 = (X0 + w - 1 + L->goff_x) / L->bsx;
    const int gmin = L->gmin, gmax = L->gmax;

    for (int by = by0; by <= by1; by++) {
        const int py0 = by * L->bsy - L->goff_y;
        const int ys = imax(Y0, py0), ye = imin(Y0 + h, py0 + L->bsy);
        const int has_t = L->overlap && by > 0;
        for (int bx = bx0; bx <= bx1; bx++) {
            const int px0 = bx * L->bsx - L->goff_x;
            const int xs = imax(X0, px0), xe = imin(X0 + w, px0 + L->bsx);
            const int has_l = L->overlap && bx > 0;
            FGBlk cur, left, top, tl;
            blk_get(&cur, L, by, bx);
            if (has_l) blk_get(&left, L, by, bx - 1);
            if (has_t) blk_get(&top, L, by - 1, bx);
            if (has_l && has_t) blk_get(&tl, L, by - 1, bx - 1);
            const int xo = has_l ? imin(xe, px0 + L->ovx) : xs;

            for (int y = ys; y < ye; y++) {
                const int ly = y - py0;
                int16_t *const d = buf + (y - Y0) * bstr - X0;
                if (has_t && ly < L->ovy) {
                    for (int x = xs; x < xe; x++) {
                        const int lx = x - px0;
                        int v = blk_val(&cur, L, ly, lx);
                        int t = blk_val(&top, L, ly + L->bsy, lx);
                        if (has_l && lx < L->ovx) {
                            v = blend2(blk_val(&left, L, ly, lx + L->bsx), v,
                                       L->wx[lx], gmin, gmax);
                            t = blend2(blk_val(&tl, L, ly + L->bsy, lx + L->bsx), t,
                                       L->wx[lx], gmin, gmax);
                        }
                        d[x] = blend2(t, v, L->wy[ly], gmin, gmax);
                    }
                    continue;
                }
                int x = xs;
                for (; x < xo; x++) {
                    const int lx = x - px0;
                    d[x] = blend2(blk_val(&left, L, ly, lx + L->bsx),
                                  blk_val(&cur, L, ly, lx), L->wx[lx], gmin, gmax);
                }
                const int yy = cur.rot ? L->Sy - 1 - ly : ly;
                const entry *const tr = cur.t + yy * GRAIN_WIDTH;
                if (!cur.rot) {
                    const entry *const t0 = tr - px0;
                    if (!cur.sign) for (; x < xe; x++) d[x] = t0[x];
                    else           for (; x < xe; x++) d[x] = -t0[x];
                } else {
                    const entry *const t1 = tr + L->Sx - 1 + px0; /* t1[-x] */
                    if (!cur.sign) for (; x < xe; x++) d[x] = t1[-x];
                    else           for (; x < xe; x++) d[x] = -t1[-x];
                }
            }
        }
    }
}

static void setup_layer(FGLayer *const L, const Dav1dFGVariant *const fgv,
                        const Dav1dFilmGrainData *const data, const int pl,
                        const int layer, const int sx, const int sy
                        HIGHBD_DECL_SUFFIX)
{
    const int bitdepth_min_8 = bitdepth_from_max(bitdepth_max) - 8;
    const int grain_ctr = 128 << bitdepth_min_8;
    L->fgv = fgv;
    L->pl = pl;
    L->seed = data->seed;
    L->layer = layer;
    if (fgv->mode == DAV1D_FGMODE_DUAL) {
        const int kh = fgv->ntmpl >> 1;
        L->Kl = layer ? fgv->ntmpl - kh : kh;
        L->kbase = layer ? kh : 0;
    } else {
        L->Kl = fgv->ntmpl;
        L->kbase = 0;
    }
    L->goff_y = layer ? (FG_BLOCK_SIZE >> 1) >> sy : 0;
    L->goff_x = layer ? (FG_BLOCK_SIZE >> 1) >> sx : 0;
    L->bsy = FG_BLOCK_SIZE >> sy;
    L->bsx = FG_BLOCK_SIZE >> sx;
    L->ovy = 2 >> sy;
    L->ovx = 2 >> sx;
    L->Sy = L->bsy + L->ovy;
    L->Sx = L->bsx + L->ovx;
    L->subx = sx;
    L->suby = sy;
    L->overlap = data->overlap_flag;
    L->wx = sx ? fgv_w_sub : fgv_w_full;
    L->wy = sy ? fgv_w_sub : fgv_w_full;
    L->gmin = -grain_ctr;
    L->gmax = grain_ctr - 1;
}

/* (a + b) / sqrt(2): ((a + b) * 181 + 128) >> 8, clipped. pmulhrsw with
 * 181 << 7 computes exactly the same value. */
#if ARCH_X86_64
static inline FGV_AVX2 int mix_row_avx2(int16_t *const a, const int16_t *const b,
                                        const int w, const int gmin, const int gmax)
{
    int x = 0;
    for (; x + 16 <= w; x += 16) {
        const __m256i s = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *) (a + x)),
                                           _mm256_loadu_si256((const __m256i *) (b + x)));
        __m256i r = _mm256_mulhrs_epi16(s, _mm256_set1_epi16(181 << 7));
        r = _mm256_min_epi16(_mm256_max_epi16(r, _mm256_set1_epi16(gmin)),
                             _mm256_set1_epi16(gmax));
        _mm256_storeu_si256((__m256i *) (a + x), r);
    }
    return x;
}
#elif ARCH_AARCH64
static inline int mix_row_neon(int16_t *const a, const int16_t *const b,
                               const int w, const int gmin, const int gmax)
{
    const int16x8_t mn = vdupq_n_s16(gmin), mx = vdupq_n_s16(gmax);
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        const int16x8_t sum = vaddq_s16(vld1q_s16(a + x), vld1q_s16(b + x));
        vst1q_s16(a + x, vminq_s16(vmaxq_s16(vqrdmulhq_n_s16(sum, 181 << 7), mn), mx));
    }
    return x;
}
#endif

static ALWAYS_INLINE void mix_layers(int16_t *const ga, const int16_t *const gb,
                                     const ptrdiff_t gstride, const int w, const int h,
                                     const int gmin, const int gmax, const int simd)
{
    for (int y = 0; y < h; y++) {
        int16_t *const a = ga + y * gstride;
        const int16_t *const b = gb + y * gstride;
        int x = 0;
#if ARCH_X86_64
        if (simd) x = mix_row_avx2(a, b, w, gmin, gmax);
#elif ARCH_AARCH64
        if (simd) x = mix_row_neon(a, b, w, gmin, gmax);
#endif
        for (; x < w; x++)
            a[x] = iclip(((a[x] + b[x]) * 181 + 128) >> 8, gmin, gmax);
    }
}

/* one band of one plane: fill the layer(s) over the whole band width, mix,
 * then scale + add (long rows keep the copy/apply loops vectorised) */
static ALWAYS_INLINE void
fgv_layers_plane(Dav1dFGVariant *const fgv,
                 pixel *const dst_row, const pixel *const src_row,
                 const ptrdiff_t stride, const int pw, const int bh, const int Y0,
                 const FGLayer *const LA, const FGLayer *const LB,
                 const int is_uv, const pixel *const luma_row,
                 const ptrdiff_t luma_stride, const int sy,
                 const FGApply *const A, const int simd HIGHBD_DECL_SUFFIX)
{
    const ptrdiff_t gs = (pw + 31) & ~31;
    int slot;
    int16_t *const ga = dav1d_fgv_scratch_get(fgv, sizeof(int16_t) * gs * FG_BLOCK_SIZE * 2 + 64,
                                              &slot);
    if (!ga) return;
    int16_t *const gb = ga + gs * FG_BLOCK_SIZE;
    layer_fill(ga, gs, Y0, 0, bh, pw, LA);
    if (LB) {
        layer_fill(gb, gs, Y0, 0, bh, pw, LB);
        mix_layers(ga, gb, gs, pw, bh, LA->gmin, LA->gmax, simd);
    }
    if (!is_uv)
        apply_y_rows(dst_row, src_row, stride, ga, gs, pw, bh, A, simd);
    else
        apply_uv_rows(dst_row, src_row, stride, luma_row, luma_stride, sy,
                      ga, gs, pw, bh, A, simd HIGHBD_TAIL_SUFFIX);
    dav1d_fgv_scratch_put(fgv, slot, ga);
}

#include "src/fg_ar_skew.h"

/* ------------------------------------------------------------------------ */
/* full-frame AR                                                              */

#if ARCH_X86_64
/* 16 samples per iteration, one hash per sample pair:
 * (h >> 5) & 0x7ff for even x, h >> 21 for odd x (x0 must be even) */
static inline FGV_AVX2 int noise_row_avx2(int16_t *const dst, const uint32_t rk,
                                          const int x0, const int n,
                                          const int32_t *const gtab)
{
    const __m256i K = _mm256_set1_epi32((int) 0x85ebca6bu);
    const __m256i m1 = _mm256_set1_epi32(0x7feb352d);
    const __m256i m2 = _mm256_set1_epi32((int) 0x846ca68bu);
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i vrk = _mm256_set1_epi32((int) rk);
    const __m256i m11 = _mm256_set1_epi32(0x7ff), m16 = _mm256_set1_epi32(0xffff);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i j = _mm256_add_epi32(_mm256_set1_epi32((x0 + i) >> 1), lane);
        __m256i h = _mm256_add_epi32(vrk, _mm256_mullo_epi32(j, K));
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
        h = _mm256_mullo_epi32(h, m1);
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 15));
        h = _mm256_mullo_epi32(h, m2);
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
        const __m256i e = _mm256_i32gather_epi32((const int *) gtab,
                              _mm256_and_si256(_mm256_srli_epi32(h, 5), m11), 4);
        const __m256i o = _mm256_i32gather_epi32((const int *) gtab,
                              _mm256_srli_epi32(h, 21), 4);
        _mm256_storeu_si256((__m256i *) (dst + i),
                            _mm256_or_si256(_mm256_and_si256(e, m16),
                                            _mm256_slli_epi32(o, 16)));
    }
    return i;
}
#endif

static ALWAYS_INLINE void noise_row(int16_t *const dst, const uint32_t key, const int y,
                                    const int x0, const int n,
                                    const int32_t *const gtab, const int simd)
{
    const uint32_t rk = dav1d_fgv_hash(key ^ dav1d_fgv_hash((uint32_t) y * 0x9E3779B9u +
                                                            0x7f4a7c15u));
    int i = 0;
#if ARCH_X86_64
    if (simd && !(x0 & 1)) i = noise_row_avx2(dst, rk, x0, n, gtab);
#endif
    for (; i < n; i++) {
        const int x = x0 + i;
        const uint32_t h = dav1d_fgv_hash(rk + (uint32_t) (x >> 1) * 0x85ebca6bu);
        const int v = (x & 1) ? (int) (h >> 21) : (int) ((h >> 5) & 0x7ff);
        dst[i] = (int16_t) gtab[v];
    }
}

#if ARCH_X86_64
/* previous-row taps for one 16-column chunk, two taps per pmaddwd */
static inline FGV_AVX2 void vtaps16_avx2(int32_t *const acc, const int16_t *const p,
                                         const ptrdiff_t *const toff,
                                         const int32_t *const cabs, const int ntap)
{
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
    for (int t = 0; t < ntap; t += 2) {
        const __m256i cab = _mm256_set1_epi32(cabs[t >> 1]);
        const __m256i va = _mm256_loadu_si256((const __m256i *) (p + toff[t]));
        const __m256i vb = _mm256_loadu_si256((const __m256i *) (p + toff[t + 1]));
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(va, vb), cab));
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(va, vb), cab));
    }
    _mm256_storeu_si256((__m256i *) acc, _mm256_permute2x128_si256(lo, hi, 0x20));
    _mm256_storeu_si256((__m256i *) (acc + 8), _mm256_permute2x128_si256(lo, hi, 0x31));
}

/* chroma-from-luma term for 4:2:x, 16 columns */
static inline FGV_AVX2 void lumaterm16_avx2(int32_t *const acc, const int16_t *const lr,
                                            const ptrdiff_t lbw, const int subx,
                                            const int suby, const int luma_coeff)
{
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i rnd = _mm256_set1_epi32((1 << (subx + suby)) >> 1);
    const __m256i lcv = _mm256_set1_epi32(luma_coeff);
    for (int c = 0; c < 16; c += 8) {
        __m256i s = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) (lr + 2 * c)), ones);
        if (suby)
            s = _mm256_add_epi32(s, _mm256_madd_epi16(
                    _mm256_loadu_si256((const __m256i *) (lr + lbw + 2 * c)), ones));
        s = _mm256_srai_epi32(_mm256_add_epi32(s, rnd), subx + suby);
        const __m256i a = _mm256_loadu_si256((const __m256i *) (acc + c));
        _mm256_storeu_si256((__m256i *) (acc + c), _mm256_add_epi32(a, _mm256_mullo_epi32(s, lcv)));
    }
}
#endif

#define AR_D 4   /* rows in flight */
#define AR_L 2   /* chunk lag between consecutive rows (>= 2 for lag <= 3) */

/* AR recursion over rows [3, rows) and columns [3, bw - 3) of buf (stride bw),
 * with the optional chroma-from-luma term -- the same integer arithmetic as
 * generate_grain_{y,uv}_c.
 *
 * The in-row taps make each row a sequential recursion (~8 cycles/sample of
 * dependency latency). Rows are therefore run as a wavefront: AR_D rows are
 * in flight, each AR_L 16-column chunks behind the row above (so all of its
 * causal neighbours are final), and their recursions are interleaved for
 * instruction-level parallelism. Results are identical to a plain raster scan.
 */
static ALWAYS_INLINE void ar_band(int16_t *const buf, const int bw, const int rows,
                                  const int8_t *const coeffs, const int lag,
                                  const int ar_shift, const int gmin, const int gmax,
                                  const int16_t *const lbuf, const int lbw,
                                  const int subx, const int suby, const int luma_coeff,
                                  int32_t *const acc_unused, Dav1dFGVariant *const fgv,
                                  const int simd)
{
    uint64_t _tp = fgv->stats > 1 ? FGV_TSC() : 0;
    int ntap = 0;
    ptrdiff_t toff[26];
    int tcf[26];
    {
        const int8_t *co = coeffs;
        for (int dy = -lag; dy < 0; dy++)
            for (int dx = -lag; dx <= lag; dx++) {
                const int cf = *(co++);
                if (cf) { toff[ntap] = dy * bw + dx; tcf[ntap] = cf; ntap++; }
            }
        if (ntap & 1) { toff[ntap] = 0; tcf[ntap] = 0; ntap++; }
    }
    /* in-row coefficients for dx = -3, -2, -1 (zero where lag < 3) */
    const int8_t *const cin = coeffs + lag * (2 * lag + 1);
    const int c3 = lag >= 3 ? cin[lag - 3] : 0;
    const int c2 = lag >= 2 ? cin[lag - 2] : 0;
    const int c1 = lag >= 1 ? cin[lag - 1] : 0;
    int32_t cabs[13];
    for (int t = 0; t < ntap; t += 2)
        cabs[t >> 1] = (int32_t) ((uint32_t) (uint16_t) tcf[t] |
                                  ((uint32_t) (uint16_t) tcf[t + 1] << 16));
    const int has_luma = lbuf && luma_coeff;
    const int cstart = 3, cend = bw - 3;
    const int nch = (cend - cstart + 15) >> 4;
    ALIGN_STK_32(int32_t, acc, AR_D, [16]);
    const int rnd = (1 << ar_shift) >> 1;

    for (int r0 = 3; r0 < rows; r0 += AR_D) {
        const int nr = imin(AR_D, rows - r0);
        int16_t *rowp[AR_D];
        int p1[AR_D], p2[AR_D], p3[AR_D];
        for (int j = 0; j < nr; j++) {
            rowp[j] = buf + (r0 + j) * bw;
            p3[j] = rowp[j][0]; p2[j] = rowp[j][1]; p1[j] = rowp[j][2];
        }
        for (int k = 0; k < nch + AR_L * (nr - 1); k++) {
            int c0[AR_D], n[AR_D], nact = 0;
            for (int j = 0; j < AR_D; j++) {
                const int kc = k - AR_L * j;
                if (j >= nr || kc < 0 || kc >= nch) { n[j] = 0; c0[j] = cstart; continue; }
                c0[j] = cstart + (kc << 4);
                n[j] = imin(16, cend - c0[j]);
                nact++;
                /* previous-row taps (+ chroma-from-luma) for this chunk */
                const int16_t *const p = rowp[j] + c0[j];
#if ARCH_X86_64
                if (simd && n[j] == 16) {
                    vtaps16_avx2(acc[j], p, toff, cabs, ntap);
                } else
#endif
                {
                    /* tap-outer order so the compiler vectorises the column loop */
                    int32_t *const a = acc[j];
                    const int nn = n[j];
                    for (int i = 0; i < nn; i++) a[i] = 0;
                    for (int t = 0; t < ntap; t++) {
                        const int cf = tcf[t];
                        const int16_t *const q = p + toff[t];
                        for (int i = 0; i < nn; i++) a[i] += cf * q[i];
                    }
                }
                if (has_luma) {
                    const int16_t *const lr = lbuf + ((r0 + j) << suby) * lbw + (c0[j] << subx);
#if ARCH_X86_64
                    if (simd && subx && n[j] == 16) {
                        lumaterm16_avx2(acc[j], lr, lbw, subx, suby, luma_coeff);
                    } else
#endif
                    for (int i = 0; i < n[j]; i++) {
                        const int lc = i << subx;
                        int luma = lr[lc];
                        if (subx) luma += lr[lc + 1];
                        if (suby) {
                            luma += lr[lbw + lc];
                            if (subx) luma += lr[lbw + lc + 1];
                        }
                        acc[j][i] += fgv_round2(luma, subx + suby) * luma_coeff;
                    }
                }
            }
            /* interleaved sequential recursions */
#define AR_STEP(j, i) do { \
                const int sum_ = acc[j][i] + c3 * p3[j] + c2 * p2[j] + c1 * p1[j] + rnd; \
                int16_t *const q_ = rowp[j] + c0[j] + (i); \
                const int v_ = iclip(*q_ + (sum_ >> ar_shift), gmin, gmax); \
                *q_ = (int16_t) v_; \
                p3[j] = p2[j]; p2[j] = p1[j]; p1[j] = v_; \
            } while (0)
            if (nact == AR_D && n[0] == 16 && n[AR_D - 1] == 16) {
                for (int i = 0; i < 16; i++) {
                    AR_STEP(0, i); AR_STEP(1, i); AR_STEP(2, i); AR_STEP(3, i);
                }
            } else {
                for (int j = 0; j < AR_D; j++)
                    for (int i = 0; i < n[j]; i++)
                        AR_STEP(j, i);
            }
#undef AR_STEP
        }
        PH(5);
    }
}

static ALWAYS_INLINE void fgv_full_row(Dav1dFGVariant *const fgv, Dav1dPicture *const out,
                                       const Dav1dPicture *const in,
                                       const uint8_t scaling[3][SCALING_SIZE],
                                       const int row, const int simd)
{
    const Dav1dFilmGrainData *const data = &out->frame_hdr->film_grain.data;
#if BITDEPTH != 8
    const int bitdepth_max = (1 << out->p.bpc) - 1;
#endif
    const int bitdepth_min_8 = bitdepth_from_max(bitdepth_max) - 8;
    const int grain_ctr = 128 << bitdepth_min_8;
    const int gmin = -grain_ctr, gmax = grain_ctr - 1;
    const int ss_y = in->p.layout == DAV1D_PIXEL_LAYOUT_I420;
    const int ss_x = in->p.layout != DAV1D_PIXEL_LAYOUT_I444;
    const int has_chroma = in->p.layout != DAV1D_PIXEL_LAYOUT_I400 &&
        (data->num_uv_points[0] || data->num_uv_points[1] ||
         data->chroma_scaling_from_luma);
    const int W = out->p.w;
    const int cpw = (W + ss_x) >> ss_x;
    /* bands of G x 32 rows share one warm-up: the first row task of each
     * group synthesises the whole group, the others return immediately */
    const int G = fgv->bands;
    if (row % G) return;
    const int Y0 = row * FG_BLOCK_SIZE;
    const int bh = imin(out->p.h - Y0, FG_BLOCK_SIZE * G);
    const int bhc = (bh + ss_y) >> ss_y;
    const int P = fgv->warmup;
    const int Pc = P >> ss_y;
    const int lag = data->ar_coeff_lag;
    const int is_id = out->seq_hdr->mtrx == DAV1D_MC_IDENTITY;

    const int lbw = FGV_XL + W + FGV_XR;
    const int lrows = P + imax(bh, has_chroma ? bhc << ss_y : 0);
    const int cbw = (FGV_XL >> ss_x) + cpw + (FGV_XR >> ss_x);
    const int crows = Pc + bhc;
    const uint32_t ykey = dav1d_fgv_hash(data->seed * 0x10001u + 0x2545F491u);
    pixel *const luma_src =
        ((pixel *) in->data[0]) + Y0 * PXSTRIDE(in->stride[0]);
    uint64_t _tp = fgv->stats > 1 ? FGV_TSC() : 0;
    FGApply A;
    int slot;
    uint8_t *scratch;
    /* field pointers: (band row 0, column 0) and strides */
    const int16_t *lf = NULL, *cf = NULL;
    ptrdiff_t lfs = 0, cfs = 0;
    int16_t *lbuf = NULL, *cbuf = NULL, *skb = NULL, *ltb = NULL;
    int ltdone = 0;
    int32_t *acc = NULL;
    SkewDims ld, cd;

    if (simd) {
        skew_dims(&ld, lbw, lrows);
        skew_dims(&cd, cbw, crows);
        const size_t nsk = imax((int) ld.skew_elems, (int) cd.skew_elems);
        const size_t need = sizeof(int16_t) * (nsk + ld.out_elems + 2 * cd.out_elems + 96);
        scratch = dav1d_fgv_scratch_get(fgv, need, &slot);
        if (!scratch) return;
        skb = (int16_t *) scratch;
        lbuf = skb + nsk + 16;
        cbuf = lbuf + ld.out_elems + 16;
        ltb = cbuf + cd.out_elems + 16;
        lf = lbuf + SK_PADL;
        lfs = ld.ostr;
        cf = cbuf + SK_PADL;
        cfs = cd.ostr;
    } else {
        const size_t need = sizeof(int16_t) * ((size_t) lrows * lbw + (size_t) crows * cbw + 32) +
                            sizeof(int32_t) * (size_t) (imax(lbw, cbw) + 16) + 64;
        scratch = dav1d_fgv_scratch_get(fgv, need, &slot);
        if (!scratch) return;
        lbuf = (int16_t *) scratch;
        cbuf = lbuf + (size_t) lrows * lbw + 16;
        acc = (int32_t *) (((uintptr_t) (cbuf + (size_t) crows * cbw + 16) + 31) & ~(uintptr_t) 31);
        lf = lbuf;
        lfs = lbw;
        cf = cbuf;
        cfs = cbw;
    }

    if (data->num_y_points) {
#if ARCH_X86_64
        if (simd) {
            ar_band_skew_avx2(lbuf, skb, &ld, ykey, Y0 - P, -FGV_XL, fgv->gtab,
                              data->ar_coeffs_y, lag, (int) data->ar_coeff_shift, gmin, gmax,
                              NULL, 0);
        } else
#elif ARCH_AARCH64
        if (simd) {
            ar_band_skew_neon(lbuf, skb, &ld, ykey, Y0 - P, -FGV_XL, fgv->gtab,
                              data->ar_coeffs_y, lag, (int) data->ar_coeff_shift, gmin, gmax,
                              NULL, 0);
        } else
#endif
        {
            for (int r = 0; r < lrows; r++)
                noise_row(lbuf + r * lbw, ykey, Y0 - P + r, -FGV_XL, lbw, fgv->gtab, 0);
            PH(0);
            ar_band(lbuf, lbw, lrows, data->ar_coeffs_y, lag, (int) data->ar_coeff_shift,
                    gmin, gmax, NULL, 0, 0, 0, 0, acc, fgv, 0);
        }
        PH(1);
        setup_apply(&A, fgv, data, scaling[0], 0, 0, 0, is_id, 0 HIGHBD_TAIL_SUFFIX);
        const ptrdiff_t stride = out->stride[0];
        apply_y_rows(((pixel *) out->data[0]) + Y0 * PXSTRIDE(stride), luma_src, stride,
                     lf + P * lfs + FGV_XL, lfs, W, bh, &A, simd);
        PH(3);
    }

    if (has_chroma) {
        if (out->p.w & ss_x) {
            pixel *ptr = luma_src;
            for (int y = 0; y < bhc; y++) {
                ptr[out->p.w] = ptr[out->p.w - 1];
                ptr += PXSTRIDE(in->stride[0]) << ss_y;
            }
        }
        const ptrdiff_t stride = out->stride[1];
        const ptrdiff_t uv_off = Y0 * PXSTRIDE(stride) >> ss_y;
        const int yc0 = Y0 >> ss_y;
        for (int pl = 0; pl < 2; pl++) {
            if (!data->chroma_scaling_from_luma && !data->num_uv_points[pl]) continue;
            const int scpl = data->chroma_scaling_from_luma ? 0 : 1 + pl;
            const uint32_t key = dav1d_fgv_hash((data->seed ^ (pl ? 0x49d8u : 0xb524u)) *
                                                0x10001u + 0x2545F491u);
            const int lc = data->num_y_points ? data->ar_coeffs_uv[pl][lag * (2 * lag + 1) + lag] : 0;
#if ARCH_X86_64
            if (simd) {
                if (lc && !ltdone) {
                    lumaterm_rows_avx2(ltb, cd.ostr, lf, lfs, crows, cbw, ss_x, ss_y);
                    ltdone = 1;
                }
                ar_band_skew_avx2(cbuf, skb, &cd, key, yc0 - Pc, -(FGV_XL >> ss_x), fgv->gtab,
                                  data->ar_coeffs_uv[pl], lag, (int) data->ar_coeff_shift, gmin, gmax,
                                  lc ? ltb : NULL, lc);
            } else
#elif ARCH_AARCH64
            if (simd) {
                if (lc && !ltdone) {
                    lumaterm_rows_neon(ltb, cd.ostr, lf, lfs, crows, cbw, ss_x, ss_y);
                    ltdone = 1;
                }
                ar_band_skew_neon(cbuf, skb, &cd, key, yc0 - Pc, -(FGV_XL >> ss_x), fgv->gtab,
                                  data->ar_coeffs_uv[pl], lag, (int) data->ar_coeff_shift, gmin, gmax,
                                  lc ? ltb : NULL, lc);
            } else
#endif
            {
                for (int r = 0; r < crows; r++)
                    noise_row(cbuf + r * cbw, key, yc0 - Pc + r, -(FGV_XL >> ss_x), cbw,
                              fgv->gtab, 0);
                PH(0);
                ar_band(cbuf, cbw, crows, data->ar_coeffs_uv[pl], lag, (int) data->ar_coeff_shift,
                        gmin, gmax, data->num_y_points ? lbuf : NULL, lbw, ss_x, ss_y, lc,
                        acc, fgv, 0);
            }
            PH(1);
            setup_apply(&A, fgv, data, scaling[scpl], scpl, 1, pl, is_id, ss_x
                        HIGHBD_TAIL_SUFFIX);
            apply_uv_rows(((pixel *) out->data[1 + pl]) + uv_off,
                          ((const pixel *) in->data[1 + pl]) + uv_off, stride,
                          luma_src, in->stride[0], ss_y,
                          cf + Pc * cfs + (FGV_XL >> ss_x), cfs, cpw, bhc, &A, simd
                          HIGHBD_TAIL_SUFFIX);
            PH(3);
        }
    }
    dav1d_fgv_scratch_put(fgv, slot, scratch);
}

/* ------------------------------------------------------------------------ */

static ALWAYS_INLINE void fgv_apply_row_impl(Dav1dFGVariant *const fgv,
                                             Dav1dPicture *const out,
                                             const Dav1dPicture *const in,
                                             const uint8_t scaling[3][SCALING_SIZE],
                                             const int row, const int simd)
{
    if (fgv->mode == DAV1D_FGMODE_FULL) {
        fgv_full_row(fgv, out, in, scaling, row, simd);
        return;
    }
    const Dav1dFilmGrainData *const data = &out->frame_hdr->film_grain.data;
    const int ss_y = in->p.layout == DAV1D_PIXEL_LAYOUT_I420;
    const int ss_x = in->p.layout != DAV1D_PIXEL_LAYOUT_I444;
    const int cpw = (out->p.w + ss_x) >> ss_x;
    const int is_id = out->seq_hdr->mtrx == DAV1D_MC_IDENTITY;
    const int dual = fgv->mode == DAV1D_FGMODE_DUAL;
    pixel *const luma_src =
        ((pixel *) in->data[0]) + row * FG_BLOCK_SIZE * PXSTRIDE(in->stride[0]);
#if BITDEPTH != 8
    const int bitdepth_max = (1 << out->p.bpc) - 1;
#endif
    FGLayer LA, LB;
    FGApply A;

    if (data->num_y_points) {
        const int bh = imin(out->p.h - row * FG_BLOCK_SIZE, FG_BLOCK_SIZE);
        setup_layer(&LA, fgv, data, 0, 0, 0, 0 HIGHBD_TAIL_SUFFIX);
        if (dual) setup_layer(&LB, fgv, data, 0, 1, 0, 0 HIGHBD_TAIL_SUFFIX);
        setup_apply(&A, fgv, data, scaling[0], 0, 0, 0, is_id, 0 HIGHBD_TAIL_SUFFIX);
        fgv_layers_plane(fgv, ((pixel *) out->data[0]) + row * FG_BLOCK_SIZE * PXSTRIDE(out->stride[0]),
                         luma_src, out->stride[0], out->p.w, bh, row * FG_BLOCK_SIZE,
                         &LA, dual ? &LB : NULL, 0, NULL, 0, 0, &A, simd
                         HIGHBD_TAIL_SUFFIX);
    }

    if (in->p.layout == DAV1D_PIXEL_LAYOUT_I400 ||
        (!data->num_uv_points[0] && !data->num_uv_points[1] &&
         !data->chroma_scaling_from_luma))
        return;

    const int bh = (imin(out->p.h - row * FG_BLOCK_SIZE, FG_BLOCK_SIZE) + ss_y) >> ss_y;

    if (out->p.w & ss_x) {
        pixel *ptr = luma_src;
        for (int y = 0; y < bh; y++) {
            ptr[out->p.w] = ptr[out->p.w - 1];
            ptr += PXSTRIDE(in->stride[0]) << ss_y;
        }
    }

    const ptrdiff_t uv_off = row * FG_BLOCK_SIZE * PXSTRIDE(out->stride[1]) >> ss_y;
    for (int pl = 0; pl < 2; pl++) {
        if (!data->chroma_scaling_from_luma && !data->num_uv_points[pl]) continue;
        const int scpl = data->chroma_scaling_from_luma ? 0 : 1 + pl;
        setup_layer(&LA, fgv, data, 1 + pl, 0, ss_x, ss_y HIGHBD_TAIL_SUFFIX);
        if (dual) setup_layer(&LB, fgv, data, 1 + pl, 1, ss_x, ss_y HIGHBD_TAIL_SUFFIX);
        setup_apply(&A, fgv, data, scaling[scpl], scpl, 1, pl, is_id, ss_x
                    HIGHBD_TAIL_SUFFIX);
        fgv_layers_plane(fgv, ((pixel *) out->data[1 + pl]) + uv_off,
                         ((const pixel *) in->data[1 + pl]) + uv_off,
                         in->stride[1], cpw, bh, (row * FG_BLOCK_SIZE) >> ss_y,
                         &LA, dual ? &LB : NULL, 1, luma_src, in->stride[0], ss_y,
                         &A, simd HIGHBD_TAIL_SUFFIX);
    }
}

static NOINLINE void fgv_apply_row_c(Dav1dFGVariant *const fgv, Dav1dPicture *const out,
                                     const Dav1dPicture *const in,
                                     const uint8_t scaling[3][SCALING_SIZE],
                                     const int row)
{
    fgv_apply_row_impl(fgv, out, in, scaling, row, 0);
}

#if ARCH_AARCH64
static NOINLINE void fgv_apply_row_neon(Dav1dFGVariant *const fgv, Dav1dPicture *const out,
                                        const Dav1dPicture *const in,
                                        const uint8_t scaling[3][SCALING_SIZE],
                                        const int row)
{
    fgv_apply_row_impl(fgv, out, in, scaling, row, 1);
}
#endif

#if ARCH_X86_64
static NOINLINE FGV_AVX2 void fgv_apply_row_avx2(Dav1dFGVariant *const fgv,
                                                 Dav1dPicture *const out,
                                                 const Dav1dPicture *const in,
                                                 const uint8_t scaling[3][SCALING_SIZE],
                                                 const int row)
{
    fgv_apply_row_impl(fgv, out, in, scaling, row, 1);
}
#endif

void bitfn(dav1d_fgv_apply_row)(Dav1dFGVariant *const fgv, Dav1dPicture *const out,
                                const Dav1dPicture *const in,
                                const uint8_t scaling[3][SCALING_SIZE], const int row)
{
#if ARCH_X86_64
    if (fgv->simd && fgv->lut32) {
        fgv_apply_row_avx2(fgv, out, in, scaling, row);
        return;
    }
#elif ARCH_AARCH64
    if (fgv->simd) {
        fgv_apply_row_neon(fgv, out, in, scaling, row);
        return;
    }
#endif
    fgv_apply_row_c(fgv, out, in, scaling, row);
}
