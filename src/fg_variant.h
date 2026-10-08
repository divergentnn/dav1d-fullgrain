/*
 * Film grain synthesis variants (NON-CONFORMANT, personal-playback fork).
 *
 * The AV1 spec (7.18.3.5) fills every frame with 32x32 crops taken at one of
 * 256 offsets from a single 82x73 grain template, so the same texture repeats
 * hundreds of times per frame. The modes below keep every signalled statistic
 * (AR model, Gaussian table, grain_scale_shift, scaling LUTs, chroma-from-luma
 * coefficients, overlap weights, clipping) but break the repetition:
 *
 *   standard   : unmodified dav1d (asm where available)            [default]
 *   standard_c : the same algorithm through the C reference only (cost ref)
 *   multi      : K templates per frame (derived seeds, generated with the
 *                normal (asm) template generator) + per-block sign flip and
 *                180-degree rotation, both of which leave the autocorrelation
 *                of a real stationary field unchanged
 *   dual       : two block layers, the second on a grid shifted by half a
 *                block (16,16); grain = (A + B) / sqrt(2); layer A and B draw
 *                from disjoint template sets so they are independent
 *   full       : full-frame AR synthesis: the signalled AR filter runs over
 *                the whole frame (per 32-row band with a warm-up), driven by
 *                hashed white noise -- no blocks, no seams, no repeats
 *
 * Runtime switches (read once per dav1d_open):
 *   DAV1D_GRAIN_MODE=standard|standard_c|multi|dual|full
 *   DAV1D_GRAIN_TEMPLATES=K   (multi/dual; default 16, 2..64)
 *   DAV1D_GRAIN_WARMUP=P      (full; default 16 rows, even, 4..64)
 *   DAV1D_GRAIN_BANDS=G       (full; 32-row bands per warm-up, default 4, 1..8)
 *   DAV1D_GRAIN_STATS=1       (print grain CPU time per frame at close)
 */

#ifndef DAV1D_SRC_FG_VARIANT_H
#define DAV1D_SRC_FG_VARIANT_H

#include <stddef.h>
#include <stdint.h>

#include "common/attributes.h"
#include <stdatomic.h>

enum Dav1dFGMode {
    DAV1D_FGMODE_STANDARD = 0,
    DAV1D_FGMODE_STANDARD_C,
    DAV1D_FGMODE_MULTI,
    DAV1D_FGMODE_DUAL,
    DAV1D_FGMODE_FULL,
};

#define FGV_MAX_TMPL 64
#define FGV_MAX_SLOTS 64

typedef struct Dav1dFGVariant {
    enum Dav1dFGMode mode;
    int ntmpl;
    int warmup;
    int bands;             /* full mode: 32-row bands per warm-up */
    int stats;
    int avx2;              /* use the AVX2 kernels (x86-64, from dav1d's cpu flags) */
    void *tmpl;            /* ntmpl template sets, each [3][GRAIN_HEIGHT+1][GRAIN_WIDTH] */
    int32_t *lut32;        /* [3][4096] scaling << (15 - scaling_shift), AVX2 path */
    int32_t gtab[2048];    /* Gaussian table >> (gaussian shift), full mode */
    size_t tmpl_set_bytes; /* per-set stride (64-byte aligned), sized for 16 bpc */
    /* scratch buffers for the full-frame mode, one per concurrently running row */
    atomic_uint_fast64_t busy;
    void *slot[FGV_MAX_SLOTS];
    size_t slot_sz[FGV_MAX_SLOTS];
    /* statistics */
    atomic_uint_fast64_t ns_prep, ns_rows, frames, rows;
    atomic_uint_fast64_t phase[8]; /* rdtsc cycles per phase (stats >= 2) */
} Dav1dFGVariant;

void dav1d_fgv_init(Dav1dFGVariant *fgv);
void dav1d_fgv_close(Dav1dFGVariant *fgv);
void *dav1d_fgv_scratch_get(Dav1dFGVariant *fgv, size_t sz, int *slot);
void dav1d_fgv_scratch_put(Dav1dFGVariant *fgv, int slot, void *buf);
uint64_t dav1d_fgv_thread_ns(void);
const char *dav1d_fgv_mode_name(enum Dav1dFGMode mode);

static inline uint32_t dav1d_fgv_hash(uint32_t x) {
    /* lowbias32 */
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline unsigned dav1d_fgv_tmpl_seed(const unsigned seed, const int k) {
    if (!k) return seed & 0xFFFF;
    const unsigned s = dav1d_fgv_hash((seed & 0xFFFF) * 0x10001u +
                                      (uint32_t) k * 0x9E3779B9u) & 0xFFFF;
    return s ? s : 1;
}

static inline uint32_t dav1d_fgv_blk_hash(const unsigned seed, const int layer,
                                          const int r, const int c)
{
    const uint32_t h0 = dav1d_fgv_hash(((seed & 0xFFFF) | ((unsigned) layer << 16)) ^
                                       0x5bd1e995u);
    const uint32_t h1 = dav1d_fgv_hash(h0 + (uint32_t) r);
    return dav1d_fgv_hash(h1 + (uint32_t) c * 0x9E3779B9u);
}

#endif /* DAV1D_SRC_FG_VARIANT_H */
