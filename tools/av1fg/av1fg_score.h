/*
 * av1fg_score: how bad a grain seed looks under the signalled parameters.
 *
 * Standard AV1 grain covers a frame with 32x32 crops of one template, at
 * offsets drawn from the seed (spec 7.18.3.5). From the seed's templates and
 * its block offsets for the actual frame size, this predicts the frame's
 * grain statistics, without rendering it:
 *   L4, L8, L16  variance of the 4/8/16-px box means relative to the grain
 *                variance: blotches (low-frequency excess)
 *   V8, V16      coefficient of variation of the local variance in 8/16-px
 *                windows: voids and clumps
 *   chroma       L8 and V8 of Cb and Cr
 *   ident        neighbouring blocks with the same offset (doubled crops)
 * badness = weighted sum of their z-scores against random seeds under the
 * same parameters (0 = typical, higher = worse).
 *
 * All arithmetic is IEEE add/multiply/divide/sqrt in a fixed order, so the
 * scores, and the seeds chosen from them, are the same on every platform.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#ifndef AV1FG_SCORE_H
#define AV1FG_SCORE_H

#include "av1fg_grain.h"

enum {
    AV1FG_M_L4, AV1FG_M_L8, AV1FG_M_L16, AV1FG_M_V8, AV1FG_M_V16,
    AV1FG_M_CB_L8, AV1FG_M_CB_V8, AV1FG_M_CR_L8, AV1FG_M_CR_V8,
    AV1FG_M_IDENT,
    AV1FG_NM
};

extern const char *const av1fg_metric_names[AV1FG_NM];
extern const double av1fg_metric_weights[AV1FG_NM];

typedef struct Av1fgStats {
    double m[AV1FG_NM];
    int valid;              /* 0: degenerate seed (constant template or a stripe of identical offsets) */
} Av1fgStats;

/* reference distribution of the metrics under one parameter set */
#define AV1FG_NREF 64
typedef struct Av1fgNorm {
    double mean[AV1FG_NM], inv_sd[AV1FG_NM];  /* inv_sd 0: metric unused */
    double ref_b[AV1FG_NREF];                  /* badness of the reference seeds, sorted */
    int nref;
} Av1fgNorm;

typedef struct Av1fgScratch Av1fgScratch;
Av1fgScratch *av1fg_scratch_alloc(void);
void av1fg_scratch_free(Av1fgScratch *s);

/* 1 if the seed can never be used (seed 0, a constant chroma template, or a
 * stripe whose offset generator starts at state 0 for this frame height). */
int av1fg_seed_degenerate(unsigned seed, const Av1fgGrainShape *sh, int height);

void av1fg_seed_stats(Av1fgScratch *sc, const Av1fgGrainShape *sh, int width, int height, unsigned seed,
                      Av1fgStats *st);
/* the same for n seeds (faster: templates are built AV1FG_BATCH at a time) */
void av1fg_seed_stats_n(Av1fgScratch *sc, const Av1fgGrainShape *sh, int width, int height, const unsigned *seeds,
                        int n, Av1fgStats *st);

/* fixed reference seeds, spread over the LFSR cycle */
extern const uint16_t av1fg_ref_seeds[AV1FG_NREF];

/* nref reference seeds (8..AV1FG_NREF, a divisor of it) */
void av1fg_norm_init(Av1fgScratch *sc, const Av1fgGrainShape *sh, int width, int height, int nref, Av1fgNorm *n);
double av1fg_badness(const Av1fgStats *st, const Av1fgNorm *n);
/* percentile of a badness among the reference seeds (0 best .. 100 worst) */
double av1fg_percentile(double b, const Av1fgNorm *n);

/* position of a 16-bit LFSR state on the 65535-step cycle (seed 0: -1) */
int av1fg_lfsr_pos(unsigned state);

#endif
