/*
 * av1fg_scan: parse every AV1 frame header of a file (any supported
 * container) and collect the film grain seeds, their file positions, the
 * distinct grain parameter sets and the display order.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#ifndef AV1FG_SCAN_H
#define AV1FG_SCAN_H

#include "av1fg_container.h"
#include "av1hdr.h"

typedef struct Av1fgParamSet {
    Av1FilmGrain fg;        /* grain_seed, apply/update flags and ref idx zeroed */
    int bit_depth, subsampling_x, subsampling_y, mono_chrome;
    int width, height;      /* frame size of the first frame using it */
    uint32_t hash;
    int frames;             /* displayed frames using it */
} Av1fgParamSet;

typedef struct Av1fgFrame {
    int64_t decode_idx;
    int64_t display_idx;    /* first display, -1 if never shown */
    int shown;              /* number of times displayed */
    int frame_type, show_frame, showable_frame, show_existing;
    int width, height, upscaled_width;
    int apply_grain, update_grain_coded, grain_ref_idx;
    uint16_t seed, new_seed;
    int param;              /* index into params, -1 without grain */
    int64_t loc;            /* index of the first seed location, -1 if none */
    int n_loc;
    int temporal_id, spatial_id;
} Av1fgFrame;

typedef struct Av1fgLoc {
    int64_t byte;           /* file offset of the byte holding the first seed bit */
    uint8_t bit;            /* bit position within it, 0 = MSB */
    uint8_t is_copy;
    int64_t frame;
} Av1fgLoc;

typedef struct Av1fgScan {
    Av1fgContainer cont;
    Av1SeqHdr seq;
    int have_seq;
    Av1fgFrame *frames;
    size_t n_frames, cap_frames;
    Av1fgLoc *locs;
    size_t n_locs, cap_locs;
    int64_t *display;       /* frame index per displayed picture, in display order */
    size_t n_display, cap_display;
    Av1fgParamSet *params;
    size_t n_params, cap_params;
    int64_t n_obus, n_headers, n_copies, n_dropped, n_show_existing, n_redundant;
    int64_t bytes_read;
    int warnings;
    char err[300];
} Av1fgScan;

typedef void (*Av1fgObuCallback)(void *ctx, const Av1Obu *obu, int64_t file_off, const Av1fgScan *s);

/* Parse the whole file. Returns 0, or -1 with a message in s->err. */
int av1fg_scan(FILE *f, int track, int strict, Av1fgScan *s, Av1fgObuCallback cb, void *cb_ctx);
void av1fg_scan_free(Av1fgScan *s);

/* One line with the grain parameters in dav1d's conventions (signed
 * coefficients), for comparisons with dav1d. */
void av1fg_format_grain(char *buf, size_t len, const Av1FilmGrain *g, int with_seed);

#endif
