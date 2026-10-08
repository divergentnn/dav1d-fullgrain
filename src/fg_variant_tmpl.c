/*
 * Film grain synthesis variants (bitdepth template). See src/fg_variant.h.
 *
 * NON-CONFORMANT: AV1 film grain synthesis is normative; these modes produce
 * different (statistically equivalent) grain than any conformant decoder.
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

#include <x86intrin.h>
#define FGV_XL 16  /* full mode: warm-up columns left of the frame */
#define PH(i) do { if (fgv->stats > 1) { const uint64_t _t = __rdtsc(); \
    atomic_fetch_add(&fgv->phase[i], _t - _tp); _tp = _t; } } while (0)
#define FGV_XR 8   /* full mode: extra columns right of the frame */

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
/* template preparation (multi / dual)                                        */

void bitfn(dav1d_fgv_prep)(const Dav1dFilmGrainDSPContext *const dsp,
                           Dav1dFGVariant *const fgv, const Dav1dPicture *const in,
                           const Dav1dFilmGrainData *const data)
{
#if BITDEPTH != 8
    const int bitdepth_max = (1 << in->p.bpc) - 1;
#endif
    if (fgv->mode != DAV1D_FGMODE_MULTI && fgv->mode != DAV1D_FGMODE_DUAL) return;
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

/* ------------------------------------------------------------------------ */
/* block layers                                                               */

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

static inline void blk_get(FGBlk *const b, const FGLayer *const L,
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

static inline int blk_val(const FGBlk *const b, const FGLayer *const L,
                          const int ly, const int lx)
{
    const int yy = b->rot ? L->Sy - 1 - ly : ly;
    const int xx = b->rot ? L->Sx - 1 - lx : lx;
    const int v = b->t[yy * GRAIN_WIDTH + xx];
    return b->sign ? -v : v;
}

static inline int blend2(const int old, const int cur, const int *const w,
                         const int gmin, const int gmax)
{
    return iclip(fgv_round2(old * w[0] + cur * w[1], 5), gmin, gmax);
}

/* Fill a w x h window at plane position (Y0, X0) with the layer's unscaled
 * grain (dav1d overlap blend at the layer's own block seams). */
static void layer_fill(int16_t *const buf, const ptrdiff_t bstr,
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

/* unscaled grain for one band (bh rows) of a plane, chunk by chunk, then the
 * standard scaling + clipping of dav1d's fgy/fguv */
static void fgv_layers_plane(pixel *const dst_row, const pixel *const src_row,
                             const ptrdiff_t stride,
                             const Dav1dFilmGrainData *const data, const int pw,
                             const int bh, const int Y0,
                             const uint8_t scaling[SCALING_SIZE],
                             const FGLayer *const LA, const FGLayer *const LB,
                             const int is_uv, const pixel *const luma_row,
                             const ptrdiff_t luma_stride, const int uv,
                             const int is_id, const int sx, const int sy
                             HIGHBD_DECL_SUFFIX)
{
    const int bitdepth_min_8 = bitdepth_from_max(bitdepth_max) - 8;
    const int gmin = LA->gmin, gmax = LA->gmax;
    int min_value, max_value;
    if (data->clip_to_restricted_range) {
        min_value = 16 << bitdepth_min_8;
        max_value = (is_uv ? (is_id ? 235 : 240) : 235) << bitdepth_min_8;
    } else {
        min_value = 0;
        max_value = BITDEPTH_MAX;
    }
    ALIGN_STK_32(int16_t, ga, FG_BLOCK_SIZE * FG_BLOCK_SIZE,);
    ALIGN_STK_32(int16_t, gb, FG_BLOCK_SIZE * FG_BLOCK_SIZE,);
    const int bsx = LA->bsx;

    for (int bx = 0; bx < pw; bx += bsx) {
        const int bw = imin(bsx, pw - bx);
        layer_fill(ga, FG_BLOCK_SIZE, Y0, bx, bh, bw, LA);
        if (LB) {
            layer_fill(gb, FG_BLOCK_SIZE, Y0, bx, bh, bw, LB);
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < bw; x++) {
                    const int i = y * FG_BLOCK_SIZE + x;
                    ga[i] = iclip(((ga[i] + gb[i]) * 181 + 128) >> 8, gmin, gmax);
                }
        }
        if (!is_uv) {
            for (int y = 0; y < bh; y++) {
                const pixel *const src = src_row + y * PXSTRIDE(stride) + bx;
                pixel *const dst = dst_row + y * PXSTRIDE(stride) + bx;
                const int16_t *const g = ga + y * FG_BLOCK_SIZE;
                for (int x = 0; x < bw; x++) {
                    const int noise = fgv_round2(scaling[src[x]] * g[x], data->scaling_shift);
                    dst[x] = iclip(src[x] + noise, min_value, max_value);
                }
            }
        } else {
            for (int y = 0; y < bh; y++) {
                const pixel *const src = src_row + y * PXSTRIDE(stride) + bx;
                pixel *const dst = dst_row + y * PXSTRIDE(stride) + bx;
                const pixel *const luma = luma_row + (y << sy) * PXSTRIDE(luma_stride) +
                                          (bx << sx);
                const int16_t *const g = ga + y * FG_BLOCK_SIZE;
                for (int x = 0; x < bw; x++) {
                    int avg = luma[x << sx];
                    if (sx) avg = (avg + luma[(x << sx) + 1] + 1) >> 1;
                    int val = avg;
                    if (!data->chroma_scaling_from_luma) {
                        const int combined = avg * data->uv_luma_mult[uv] +
                                             src[x] * data->uv_mult[uv];
                        val = iclip_pixel((combined >> 6) +
                                          (data->uv_offset[uv] * (1 << bitdepth_min_8)));
                    }
                    const int noise = fgv_round2(scaling[val] * g[x], data->scaling_shift);
                    dst[x] = iclip(src[x] + noise, min_value, max_value);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------ */
/* full-frame AR                                                              */

static void noise_row(int16_t *const dst, const uint32_t key, const int y,
                      const int x0, const int n, const int shift)
{
    const uint32_t rk = dav1d_fgv_hash(key ^ dav1d_fgv_hash((uint32_t) y * 0x9E3779B9u +
                                                            0x7f4a7c15u));
    for (int i = 0; i < n; i++) {
        const int x = x0 + i;
        const uint32_t h = dav1d_fgv_hash(rk + (uint32_t) (x >> 1) * 0x85ebca6bu);
        const int v = (x & 1) ? (int) (h >> 21) : (int) ((h >> 5) & 0x7ff);
        dst[i] = (int16_t) fgv_round2(dav1d_gaussian_sequence[v], shift);
    }
}

/* AR recursion over rows [3, rows) and columns [3, bw - 3) of buf (stride bw),
 * with the optional chroma-from-luma term. The AR is causal, so the
 * contribution of the previous `lag` rows is gathered first (vectorisable)
 * and only the in-row taps remain sequential. */
static void ar_band(int16_t *const buf, const int bw, const int rows,
                    const int8_t *const coeffs, const int lag, const int ar_shift,
                    const int gmin, const int gmax,
                    const int16_t *const lbuf, const int lbw, const int subx,
                    const int suby, const int luma_coeff, int32_t *const acc,
                    Dav1dFGVariant *const fgv)
{
    uint64_t _tp = fgv->stats > 1 ? __rdtsc() : 0;
    for (int r = 3; r < rows; r++) {
        int16_t *const row = buf + r * bw;
        const int8_t *co = coeffs;
        for (int c = 3; c < bw - 3; c++) acc[c] = 0;
        for (int dy = -lag; dy < 0; dy++) {
            const int16_t *const rr = row + dy * bw;
            for (int dx = -lag; dx <= lag; dx++) {
                const int cf = *(co++);
                if (!cf) continue;
                for (int c = 3; c < bw - 3; c++)
                    acc[c] += cf * rr[c + dx];
            }
        }
        if (lbuf && luma_coeff) {
            const int16_t *const lr = lbuf + (r << suby) * lbw;
            for (int c = 3; c < bw - 3; c++) {
                const int lc = c << subx;
                int luma = lr[lc];
                if (subx) luma += lr[lc + 1];
                if (suby) {
                    luma += lr[lbw + lc];
                    if (subx) luma += lr[lbw + lc + 1];
                }
                luma = fgv_round2(luma, subx + suby);
                acc[c] += luma * luma_coeff;
            }
        }
        PH(4);
        /* in-row taps: co points at the dy=0 coefficients (dx=-lag..-1) */
        if (lag == 3) {
            const int c3 = co[0], c2 = co[1], c1 = co[2];
            int p3 = row[0], p2 = row[1], p1 = row[2];
            for (int c = 3; c < bw - 3; c++) {
                const int sum = acc[c] + c3 * p3 + c2 * p2 + c1 * p1;
                const int v = iclip(row[c] + fgv_round2(sum, ar_shift), gmin, gmax);
                row[c] = (int16_t) v;
                p3 = p2; p2 = p1; p1 = v;
            }
        } else {
            for (int c = 3; c < bw - 3; c++) {
                int sum = acc[c];
                for (int dx = -lag; dx < 0; dx++)
                    sum += co[dx + lag] * row[c + dx];
                row[c] = (int16_t) iclip(row[c] + fgv_round2(sum, ar_shift), gmin, gmax);
            }
        }
        PH(5);
    }
}

static void fgv_full_row(Dav1dFGVariant *const fgv, Dav1dPicture *const out,
                         const Dav1dPicture *const in,
                         const uint8_t scaling[3][SCALING_SIZE], const int row)
{
    const Dav1dFilmGrainData *const data = &out->frame_hdr->film_grain.data;
#if BITDEPTH != 8
    const int bitdepth_max = (1 << out->p.bpc) - 1;
#endif
    const int bitdepth_min_8 = bitdepth_from_max(bitdepth_max) - 8;
    const int grain_ctr = 128 << bitdepth_min_8;
    const int gmin = -grain_ctr, gmax = grain_ctr - 1;
    const int gshift = 4 - bitdepth_min_8 + data->grain_scale_shift;
    const int ss_y = in->p.layout == DAV1D_PIXEL_LAYOUT_I420;
    const int ss_x = in->p.layout != DAV1D_PIXEL_LAYOUT_I444;
    const int has_chroma = in->p.layout != DAV1D_PIXEL_LAYOUT_I400 &&
        (data->num_uv_points[0] || data->num_uv_points[1] ||
         data->chroma_scaling_from_luma);
    const int W = out->p.w;
    const int cpw = (W + ss_x) >> ss_x;
    const int Y0 = row * FG_BLOCK_SIZE;
    const int bh = imin(out->p.h - Y0, FG_BLOCK_SIZE);
    const int bhc = (bh + ss_y) >> ss_y;
    const int P = fgv->warmup;
    const int Pc = P >> ss_y;
    const int lag = data->ar_coeff_lag;
    const int is_id = out->seq_hdr->mtrx == DAV1D_MC_IDENTITY;

    const int lbw = FGV_XL + W + FGV_XR;
    const int lrows = P + imax(bh, has_chroma ? bhc << ss_y : 0);
    const int cbw = (FGV_XL >> ss_x) + cpw + (FGV_XR >> ss_x);
    const int crows = Pc + bhc;
    const size_t need = sizeof(int16_t) * ((size_t) lrows * lbw + (size_t) crows * cbw) +
                        sizeof(int32_t) * (size_t) imax(lbw, cbw) + 64;
    int slot;
    uint8_t *const scratch = dav1d_fgv_scratch_get(fgv, need, &slot);
    if (!scratch) return;
    int16_t *const lbuf = (int16_t *) scratch;
    int16_t *const cbuf = lbuf + (size_t) lrows * lbw;
    int32_t *const acc = (int32_t *) (((uintptr_t) (cbuf + (size_t) crows * cbw) + 15) & ~(uintptr_t) 15);

    pixel *const luma_src =
        ((pixel *) in->data[0]) + Y0 * PXSTRIDE(in->stride[0]);
    uint64_t _tp = fgv->stats > 1 ? __rdtsc() : 0;

    if (data->num_y_points) {
        const uint32_t key = dav1d_fgv_hash(data->seed * 0x10001u + 0x2545F491u);
        for (int r = 0; r < lrows; r++)
            noise_row(lbuf + r * lbw, key, Y0 - P + r, -FGV_XL, lbw, gshift);
        PH(0);
        ar_band(lbuf, lbw, lrows, data->ar_coeffs_y, lag, data->ar_coeff_shift,
                gmin, gmax, NULL, 0, 0, 0, 0, acc, fgv);
        PH(1);

        int min_value, max_value;
        if (data->clip_to_restricted_range) {
            min_value = 16 << bitdepth_min_8;
            max_value = 235 << bitdepth_min_8;
        } else {
            min_value = 0;
            max_value = BITDEPTH_MAX;
        }
        const ptrdiff_t stride = out->stride[0];
        pixel *const dst_row = ((pixel *) out->data[0]) + Y0 * PXSTRIDE(stride);
        for (int y = 0; y < bh; y++) {
            const pixel *const src = luma_src + y * PXSTRIDE(stride);
            pixel *const dst = dst_row + y * PXSTRIDE(stride);
            const int16_t *const g = lbuf + (P + y) * lbw + FGV_XL;
            for (int x = 0; x < W; x++) {
                const int noise = fgv_round2(scaling[0][src[x]] * g[x], data->scaling_shift);
                dst[x] = iclip(src[x] + noise, min_value, max_value);
            }
        }
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
        int min_value, max_value;
        if (data->clip_to_restricted_range) {
            min_value = 16 << bitdepth_min_8;
            max_value = (is_id ? 235 : 240) << bitdepth_min_8;
        } else {
            min_value = 0;
            max_value = BITDEPTH_MAX;
        }
        for (int pl = 0; pl < 2; pl++) {
            if (!data->chroma_scaling_from_luma && !data->num_uv_points[pl]) continue;
            const uint8_t *const sc = scaling[data->chroma_scaling_from_luma ? 0 : 1 + pl];
            const uint32_t key = dav1d_fgv_hash((data->seed ^ (pl ? 0x49d8u : 0xb524u)) *
                                                0x10001u + 0x2545F491u);
            for (int r = 0; r < crows; r++)
                noise_row(cbuf + r * cbw, key, yc0 - Pc + r, -(FGV_XL >> ss_x), cbw, gshift);
            PH(0);
            ar_band(cbuf, cbw, crows, data->ar_coeffs_uv[pl], lag, data->ar_coeff_shift,
                    gmin, gmax, data->num_y_points ? lbuf : NULL, lbw, ss_x, ss_y,
                    data->num_y_points ? data->ar_coeffs_uv[pl][lag * (2 * lag + 1) + lag] : 0,
                    acc, fgv);
            PH(1);
            pixel *const dst_row = ((pixel *) out->data[1 + pl]) + uv_off;
            const pixel *const src_row = ((const pixel *) in->data[1 + pl]) + uv_off;
            for (int y = 0; y < bhc; y++) {
                const pixel *const src = src_row + y * PXSTRIDE(stride);
                pixel *const dst = dst_row + y * PXSTRIDE(stride);
                const pixel *const luma = luma_src + (y << ss_y) * PXSTRIDE(in->stride[0]);
                const int16_t *const g = cbuf + (Pc + y) * cbw + (FGV_XL >> ss_x);
                for (int x = 0; x < cpw; x++) {
                    int avg = luma[x << ss_x];
                    if (ss_x) avg = (avg + luma[(x << ss_x) + 1] + 1) >> 1;
                    int val = avg;
                    if (!data->chroma_scaling_from_luma) {
                        const int combined = avg * data->uv_luma_mult[pl] +
                                             src[x] * data->uv_mult[pl];
                        val = iclip_pixel((combined >> 6) +
                                          (data->uv_offset[pl] * (1 << bitdepth_min_8)));
                    }
                    const int noise = fgv_round2(sc[val] * g[x], data->scaling_shift);
                    dst[x] = iclip(src[x] + noise, min_value, max_value);
                }
            }
            PH(3);
        }
    }
    dav1d_fgv_scratch_put(fgv, slot, scratch);
}

/* ------------------------------------------------------------------------ */

void bitfn(dav1d_fgv_apply_row)(Dav1dFGVariant *const fgv, Dav1dPicture *const out,
                                const Dav1dPicture *const in,
                                const uint8_t scaling[3][SCALING_SIZE], const int row)
{
    if (fgv->mode == DAV1D_FGMODE_FULL) {
        fgv_full_row(fgv, out, in, scaling, row);
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

    if (data->num_y_points) {
        const int bh = imin(out->p.h - row * FG_BLOCK_SIZE, FG_BLOCK_SIZE);
        setup_layer(&LA, fgv, data, 0, 0, 0, 0 HIGHBD_TAIL_SUFFIX);
        if (dual) setup_layer(&LB, fgv, data, 0, 1, 0, 0 HIGHBD_TAIL_SUFFIX);
        fgv_layers_plane(((pixel *) out->data[0]) + row * FG_BLOCK_SIZE * PXSTRIDE(out->stride[0]),
                         luma_src, out->stride[0], data, out->p.w, bh,
                         row * FG_BLOCK_SIZE, scaling[0], &LA, dual ? &LB : NULL,
                         0, NULL, 0, 0, is_id, 0, 0 HIGHBD_TAIL_SUFFIX);
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
        setup_layer(&LA, fgv, data, 1 + pl, 0, ss_x, ss_y HIGHBD_TAIL_SUFFIX);
        if (dual) setup_layer(&LB, fgv, data, 1 + pl, 1, ss_x, ss_y HIGHBD_TAIL_SUFFIX);
        fgv_layers_plane(((pixel *) out->data[1 + pl]) + uv_off,
                         ((const pixel *) in->data[1 + pl]) + uv_off,
                         in->stride[1], data, cpw, bh,
                         (row * FG_BLOCK_SIZE) >> ss_y,
                         scaling[data->chroma_scaling_from_luma ? 0 : 1 + pl],
                         &LA, dual ? &LB : NULL, 1, luma_src, in->stride[0], pl,
                         is_id, ss_x, ss_y HIGHBD_TAIL_SUFFIX);
    }
}
