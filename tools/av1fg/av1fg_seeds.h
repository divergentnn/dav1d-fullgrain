/*
 * av1fg_seeds: rate a file's grain seeds and choose new ones.
 *
 * Seed policy (deterministic, the same on every platform):
 *  - pool: 2048 seeds that score well under the grain parameters of 11 real
 *    files (SVT-AV1, libaom, rav1e; 720p to 4K), at most 64 per 2048
 *    positions of the 65535-step LFSR cycle (neighbouring LFSR states share
 *    most of their template, so good seeds come in clusters); av1fg_pool.inc,
 *    built with --rank-all and tests/mkpool.py, sorted by LFSR position;
 *  - each displayed frame takes the next `candidates` pool entries (default
 *    15) of a walk with a stride of about 0.618 x the pool size, and keeps
 *    the one with the lowest badness under its own grain parameters and frame
 *    size, skipping seeds that are degenerate for it or whose luma or chroma
 *    template would share more than half its samples with the template of
 *    one of the previous `overlap_window` (3) displayed frames; the file's
 *    own seed is kept when it is at least as good;
 *  - the walk visits every pool entry once per 2048 / candidates displayed
 *    frames (136 by default, 5.7 s at 24 fps), so every frame gets a new
 *    seed, there are no short cycles, and no seed is used twice in a row.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#ifndef AV1FG_SEEDS_H
#define AV1FG_SEEDS_H

#include <stdio.h>

#include "av1fg_scan.h"
#include "av1fg_score.h"

extern const uint16_t av1fg_pool[];
extern const int av1fg_pool_size;

typedef struct Av1fgSeedOpts {
    int candidates;      /* pool seeds tried per frame */
    double bad_pct;      /* a seed is bad when worse than (100 - bad_pct) % of random seeds */
    int only_bad;        /* replace only bad seeds */
    int overlap_window;  /* displayed frames checked for template overlap */
    int max_sets;        /* above this many distinct parameter sets, use half the reference seeds */
    int threads;
    int nref;            /* reference seeds per parameter set (0: automatic) */
} Av1fgSeedOpts;

typedef struct Av1fgFrameScore {
    float b_old, p_old;  /* badness and percentile of the file's seed */
    float b_new, p_new;  /* of the chosen seed */
    int set;             /* normalisation group */
} Av1fgFrameScore;

typedef struct Av1fgSet {
    Av1fgGrainShape sh;
    int width, height;
    int64_t first_display, last_display;
    int frames;
    Av1fgNorm norm;
} Av1fgSet;

typedef struct Av1fgSeedReport {
    Av1fgFrameScore *fs;     /* per frame (index into Av1fgScan.frames) */
    Av1fgSet *sets;
    int n_sets;
    int nref;                /* reference seeds per parameter set */
    int64_t n_scored, n_changed, n_bad_old, n_bad_new, n_vbad_old, n_vbad_new;
    int64_t n_fallback;      /* frames where no candidate met the overlap rule */
    double secs;
} Av1fgSeedReport;

void av1fg_seed_opts_default(Av1fgSeedOpts *o);

/* Score the file's seeds; choose new ones (frames[].new_seed) unless list is
 * given, in which case the listed seeds are written in display order. */
int av1fg_choose_seeds(Av1fgScan *s, const Av1fgSeedOpts *o, const uint16_t *list, size_t nlist, int write,
                       Av1fgSeedReport *r);
void av1fg_print_report(FILE *out, const Av1fgScan *s, const Av1fgSeedOpts *o, const Av1fgSeedReport *r, int write);
void av1fg_report_free(Av1fgSeedReport *r);

/* --rank-all: badness of every seed 1..65535 averaged over up to max_sets
 * parameter sets of the file (z-scored per set), written as "seed badness"
 * lines. Used to build the pool. */
int av1fg_rank_all(const Av1fgScan *s, int max_sets, FILE *out);

#endif
