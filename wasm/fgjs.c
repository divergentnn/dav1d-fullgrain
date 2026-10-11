/*
 * av1-fullgrain for WebAssembly, grain only: the fork's film grain synthesis
 * (full-frame and standard AV1 grain, src/fg_*.c and src/filmgrain_tmpl.c)
 * applied to pictures that the browser decoded without grain. No decoder.
 * The player uses it where WebGPU is missing (wasm/web/grain-worker.mjs);
 * the output is bit-identical to native dav1d and to the WebGPU shaders.
 *
 *   fgjs *f = fgjs_open(threads);
 *   uint8_t *buf = fgjs_alloc(f, bytes);           // planes live in WASM memory
 *   fgjs_apply(f, mode, w, h, bpc, layout, mtrx, params, in[3], out[3], strides)
 *
 * Strides must be multiples of 64 bytes (dav1d's block loops assume it).
 * Band groups (full) / 32-row blocks (standard) are shared out to `threads`
 * pthreads.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "config.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <emscripten.h>

#include "dav1d/headers.h"
#include "dav1d/picture.h"
#include "src/fg_apply.h"
#include "src/fg_variant.h"
#include "src/filmgrain.h"

#define MAXTH 16

typedef struct fgjs {
    Dav1dFilmGrainDSPContext dsp8, dsp16;
    Dav1dFGVariant fgv;
    ALIGN(uint8_t scaling[3][4096], 64);
    ALIGN(int16_t grain_lut[3][GRAIN_HEIGHT + 1][GRAIN_WIDTH], 64);
    Dav1dPicture in, out;
    Dav1dFrameHeader fh;
    Dav1dSequenceHeader sh;
    /* thread pool */
    pthread_t th[MAXTH];
    int nth, gen, quit, busy;
    pthread_mutex_t m;
    pthread_cond_t go, done;
    atomic_int next;
    int rows;
} fgjs;

static void rows_work(fgjs *const f) {
    for (;;) {
        const int row = atomic_fetch_add(&f->next, 1);
        if (row >= f->rows) return;
        if (f->out.p.bpc == 8)
            dav1d_apply_grain_row_8bpc(&f->dsp8, &f->fgv, &f->out, &f->in, f->scaling, f->grain_lut, row);
        else
            dav1d_apply_grain_row_16bpc(&f->dsp16, &f->fgv, &f->out, &f->in, f->scaling, f->grain_lut, row);
    }
}

static void *worker(void *arg) {
    fgjs *const f = arg;
    int seen = 0;
    pthread_mutex_lock(&f->m);
    for (;;) {
        while (f->gen == seen && !f->quit) pthread_cond_wait(&f->go, &f->m);
        if (f->quit) break;
        seen = f->gen;
        pthread_mutex_unlock(&f->m);
        rows_work(f);
        pthread_mutex_lock(&f->m);
        if (!--f->busy) pthread_cond_signal(&f->done);
    }
    pthread_mutex_unlock(&f->m);
    return NULL;
}

EMSCRIPTEN_KEEPALIVE
fgjs *fgjs_open(int threads) {
    fgjs *const f = aligned_alloc(64, (sizeof(fgjs) + 63) & ~(size_t) 63);
    if (!f) return NULL;
    memset(f, 0, sizeof(*f));
    dav1d_film_grain_dsp_init_8bpc(&f->dsp8);
    dav1d_film_grain_dsp_init_16bpc(&f->dsp16);
    dav1d_fgv_init(&f->fgv);
    pthread_mutex_init(&f->m, NULL);
    pthread_cond_init(&f->go, NULL);
    pthread_cond_init(&f->done, NULL);
    if (threads > MAXTH) threads = MAXTH;
    for (int i = 0; i + 1 < threads; i++)          /* the caller is the last worker */
        if (!pthread_create(&f->th[f->nth], NULL, worker, f)) f->nth++;
    return f;
}

EMSCRIPTEN_KEEPALIVE int fgjs_threads(const fgjs *f) { return f->nth + 1; }

EMSCRIPTEN_KEEPALIVE void *fgjs_alloc(fgjs *f, int bytes) { (void) f; return aligned_alloc(64, (bytes + 63) & ~63); }
EMSCRIPTEN_KEEPALIVE void fgjs_free(void *p) { free(p); }

/* params: the Dav1dFilmGrainData fields as int32 (see grain-worker.mjs, P_* offsets):
 * seed, num_y_points, y_points[14][2], chroma_scaling_from_luma, num_uv_points[2], uv_points[2][10][2],
 * scaling_shift, ar_coeff_lag, ar_coeffs_y[24], ar_coeffs_uv[2][28], ar_coeff_shift, grain_scale_shift,
 * uv_mult[2], uv_luma_mult[2], uv_offset[2], overlap_flag, clip_to_restricted_range
 * mode: 0 standard, 1 full. Returns 0, or -1 for bad arguments. */
EMSCRIPTEN_KEEPALIVE
int fgjs_apply(fgjs *const f, const int mode, const int w, const int h, const int bpc, const int layout,
               const int mtrx, const int32_t *const pr, uint8_t *const in_y, uint8_t *const in_u,
               uint8_t *const in_v, uint8_t *const out_y, uint8_t *const out_u, uint8_t *const out_v,
               const int stride_y, const int stride_uv)
{
    if (w <= 0 || h <= 0 || (bpc != 8 && bpc != 10 && bpc != 12) || layout < 0 || layout > 3 ||
        stride_y % 64 || stride_uv % 64)
        return -1;
    Dav1dFilmGrainData *const g = &f->fh.film_grain.data;
    memset(g, 0, sizeof(*g));
    const int32_t *p = pr;
    g->seed = (unsigned) *p++ & 0xffff;
    g->num_y_points = *p++;
    for (int i = 0; i < 14; i++) { g->y_points[i][0] = *p++; g->y_points[i][1] = *p++; }
    g->chroma_scaling_from_luma = *p++;
    g->num_uv_points[0] = *p++;
    g->num_uv_points[1] = *p++;
    for (int pl = 0; pl < 2; pl++)
        for (int i = 0; i < 10; i++) { g->uv_points[pl][i][0] = *p++; g->uv_points[pl][i][1] = *p++; }
    g->scaling_shift = *p++;
    g->ar_coeff_lag = *p++;
    for (int i = 0; i < 24; i++) g->ar_coeffs_y[i] = *p++;
    for (int pl = 0; pl < 2; pl++)
        for (int i = 0; i < 28; i++) g->ar_coeffs_uv[pl][i] = *p++;
    g->ar_coeff_shift = *p++;
    g->grain_scale_shift = *p++;
    for (int pl = 0; pl < 2; pl++) g->uv_mult[pl] = *p++;
    for (int pl = 0; pl < 2; pl++) g->uv_luma_mult[pl] = *p++;
    for (int pl = 0; pl < 2; pl++) g->uv_offset[pl] = *p++;
    g->overlap_flag = *p++;
    g->clip_to_restricted_range = *p++;
    if (g->num_y_points > 14 || g->num_uv_points[0] > 10 || g->num_uv_points[1] > 10 ||
        g->ar_coeff_lag > 3 || g->scaling_shift < 8 || g->scaling_shift > 11 ||
        g->ar_coeff_shift < 6 || g->ar_coeff_shift > 9 || g->grain_scale_shift > 3)
        return -1;
    f->fh.film_grain.present = 1;
    f->fh.film_grain.update = 1;
    f->sh.mtrx = mtrx;
    f->fgv.mode = mode ? DAV1D_FGMODE_FULL : DAV1D_FGMODE_STANDARD;

    Dav1dPicture *const in = &f->in, *const out = &f->out;
    memset(in, 0, sizeof(*in));
    in->p.w = w; in->p.h = h; in->p.bpc = bpc; in->p.layout = layout;
    in->data[0] = in_y; in->data[1] = in_u; in->data[2] = in_v;
    in->stride[0] = stride_y; in->stride[1] = stride_uv;
    in->frame_hdr = &f->fh; in->seq_hdr = &f->sh;
    *out = *in;
    out->data[0] = out_y; out->data[1] = out_u; out->data[2] = out_v;

    if (bpc == 8)
        dav1d_prep_grain_8bpc(&f->dsp8, &f->fgv, out, in, f->scaling, f->grain_lut);
    else
        dav1d_prep_grain_16bpc(&f->dsp16, &f->fgv, out, in, f->scaling, f->grain_lut);
    f->rows = (h + FG_BLOCK_SIZE - 1) / FG_BLOCK_SIZE;
    atomic_store(&f->next, 0);
    pthread_mutex_lock(&f->m);
    f->busy = f->nth;
    f->gen++;
    pthread_cond_broadcast(&f->go);
    pthread_mutex_unlock(&f->m);
    rows_work(f);
    pthread_mutex_lock(&f->m);
    while (f->busy) pthread_cond_wait(&f->done, &f->m);
    pthread_mutex_unlock(&f->m);
    return 0;
}

EMSCRIPTEN_KEEPALIVE
void fgjs_close(fgjs *const f) {
    if (!f) return;
    pthread_mutex_lock(&f->m);
    f->quit = 1;
    pthread_cond_broadcast(&f->go);
    pthread_mutex_unlock(&f->m);
    for (int i = 0; i < f->nth; i++) pthread_join(f->th[i], NULL);
    dav1d_fgv_close(&f->fgv);
    free(f);
}
