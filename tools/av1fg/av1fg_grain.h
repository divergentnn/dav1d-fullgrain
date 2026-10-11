/*
 * av1fg_grain: the AV1 film grain templates of a seed (spec 7.18.3.3) and
 * the per-block offsets (7.18.3.5), bit-exact, plus the seed quality
 * measures av1-grain-seed uses.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#ifndef AV1FG_GRAIN_H
#define AV1FG_GRAIN_H

#include <stdint.h>

#include "av1hdr.h"

#define AV1FG_TPL_H 73
#define AV1FG_TPL_W 82

extern const int16_t av1fg_gaussian_sequence[2048];

typedef struct Av1fgTemplates {
    int16_t y[AV1FG_TPL_H][AV1FG_TPL_W];
    int16_t cb[AV1FG_TPL_H][AV1FG_TPL_W];
    int16_t cr[AV1FG_TPL_H][AV1FG_TPL_W];
    int cw, ch;            /* chroma template size (44x38 for 4:2:0) */
} Av1fgTemplates;

/* The grain parameters a template depends on. */
typedef struct Av1fgGrainShape {
    int bit_depth, ssx, ssy, mono;
    Av1FilmGrain fg;
} Av1fgGrainShape;

/* The AR-filtered luma and chroma templates of seed, exactly as decoders
 * build them. planes: bit 0 luma, bit 1 Cb, bit 2 Cr (chroma needs luma
 * when num_y_points > 0). */
void av1fg_templates(const Av1fgGrainShape *s, unsigned seed, int planes, Av1fgTemplates *t);

/* The same for up to AV1FG_BATCH seeds at once (lane-parallel AR filter,
 * several times faster per seed; bit-identical to av1fg_templates()). */
#define AV1FG_BATCH 8
typedef struct Av1fgBatch {
    int16_t y[AV1FG_TPL_H][AV1FG_TPL_W][AV1FG_BATCH];
    int16_t c[2][AV1FG_TPL_H][AV1FG_TPL_W][AV1FG_BATCH];
} Av1fgBatch;
void av1fg_templates_batch(const Av1fgGrainShape *s, const unsigned *seeds, int n, int planes, Av1fgBatch *b,
                           Av1fgTemplates *t);

/* White noise before the AR filter (Gaussian_Sequence values, unscaled)
 * for the luma template of seed: 73 x 82 values. */
void av1fg_white_luma(unsigned seed, int16_t w[AV1FG_TPL_H][AV1FG_TPL_W]);

/* The 8-bit random value of every 32x32 luma block (offsetX = v >> 4,
 * offsetY = v & 15) for a frame of width x height. out: rows x cols. */
void av1fg_block_offsets(unsigned seed, int width, int height, uint8_t *out, int *rows, int *cols);

#endif
