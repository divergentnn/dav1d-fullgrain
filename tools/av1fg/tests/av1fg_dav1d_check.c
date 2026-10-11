/*
 * av1fg-dav1d-check: cross-checks av1-grain-seed against libdav1d.
 *
 *   av1fg-dav1d-check params FILE
 *       one line per output picture: the film grain parameters dav1d parsed,
 *       in the format of `av1-grain-seed --dump-display`
 *   av1fg-dav1d-check edit ORIGINAL PATCHED
 *       1. both files decode to identical pictures with grain off;
 *       2. their grain parameters are identical except grain_seed;
 *       3. PATCHED decoded with grain equals ORIGINAL's grain-free picture
 *          with grain applied using PATCHED's parameters, i.e. the grain-on
 *          decodes differ only in the grain, drawn from the new seed.
 *       Prints the number of pictures, how many seeds changed, and OK/FAIL.
 *
 * Run with DAV1D_GRAIN_MODE=standard DAV1D_GRAIN_FORCE=0 on this fork.
 * FILE: IVF, Annex B or Section 5 (dav1d's demuxers).
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dav1d/dav1d.h"
#include "input/input.h"

typedef struct Dec {
    DemuxerContext *in;
    Dav1dContext *c;
    Dav1dData data;
    int eof;
} Dec;

static int dec_open(Dec *d, const char *path, int grain)
{
    memset(d, 0, sizeof(*d));
    unsigned fps[2], n, tb[2];
    if (input_open(&d->in, NULL, path, fps, &n, tb) < 0) {
        fprintf(stderr, "cannot open %s\n", path);
        return -1;
    }
    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = 1;
    s.max_frame_delay = 1;
    s.apply_grain = grain;
    if (dav1d_open(&d->c, &s) < 0)
        return -1;
    if (input_read(d->in, &d->data) < 0)
        d->eof = 1;
    return 0;
}

/* next output picture; 0 ok, 1 end, -1 error */
static int dec_next(Dec *d, Dav1dPicture *p)
{
    memset(p, 0, sizeof(*p));
    for (;;) {
        if (d->data.sz) {
            const int r = dav1d_send_data(d->c, &d->data);
            if (r < 0 && r != DAV1D_ERR(EAGAIN))
                return -1;
        }
        const int r = dav1d_get_picture(d->c, p);
        if (r == 0)
            return 0;
        if (r != DAV1D_ERR(EAGAIN))
            return -1;
        if (!d->data.sz) {
            if (d->eof) {
                const int r2 = dav1d_get_picture(d->c, p);
                return r2 == 0 ? 0 : r2 == DAV1D_ERR(EAGAIN) ? 1 : -1;
            }
            if (input_read(d->in, &d->data) < 0)
                d->eof = 1;
        }
    }
}

static void dec_close(Dec *d)
{
    dav1d_data_unref(&d->data);
    dav1d_close(&d->c);
    input_close(d->in);
}

static void print_grain(const Dav1dFrameHeader *h)
{
    const Dav1dFilmGrainData *g = &h->film_grain.data;
    if (!h->film_grain.present) {
        printf("grain 0\n");
        return;
    }
    printf("seed %u y[", g->seed);
    for (int i = 0; i < g->num_y_points; i++)
        printf("%s%d:%d", i ? " " : "", g->y_points[i][0], g->y_points[i][1]);
    printf("] csfl %d", g->chroma_scaling_from_luma);
    for (int pl = 0; pl < 2; pl++) {
        printf(" %s[", pl ? "cr" : "cb");
        for (int i = 0; i < g->num_uv_points[pl]; i++)
            printf("%s%d:%d", i ? " " : "", g->uv_points[pl][i][0], g->uv_points[pl][i][1]);
        printf("]");
    }
    printf(" scaling_shift %d lag %d", g->scaling_shift, g->ar_coeff_lag);
    const int npl = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
    const int npc = npl + (g->num_y_points ? 1 : 0);
    printf(" ary[");
    if (g->num_y_points)
        for (int i = 0; i < npl; i++)
            printf("%s%d", i ? " " : "", g->ar_coeffs_y[i]);
    for (int pl = 0; pl < 2; pl++) {
        printf("] ar%s[", pl ? "cr" : "cb");
        if (g->num_uv_points[pl] || g->chroma_scaling_from_luma)
            for (int i = 0; i < npc; i++)
                printf("%s%d", i ? " " : "", g->ar_coeffs_uv[pl][i]);
    }
    printf("] ar_shift %d gss %d", (int)g->ar_coeff_shift, g->grain_scale_shift);
    for (int pl = 0; pl < 2; pl++)
        if (g->num_uv_points[pl])
            printf(" %s %d %d %d", pl ? "cr" : "cb", g->uv_mult[pl], g->uv_luma_mult[pl], g->uv_offset[pl]);
    printf(" overlap %d clip %d\n", g->overlap_flag, g->clip_to_restricted_range);
}

static int same_pixels(const Dav1dPicture *a, const Dav1dPicture *b)
{
    if (a->p.w != b->p.w || a->p.h != b->p.h || a->p.layout != b->p.layout || a->p.bpc != b->p.bpc)
        return 0;
    const int bps = a->p.bpc > 8 ? 2 : 1;
    for (int pl = 0; pl < (a->p.layout == DAV1D_PIXEL_LAYOUT_I400 ? 1 : 3); pl++) {
        const int ssx = pl && a->p.layout != DAV1D_PIXEL_LAYOUT_I444;
        const int ssy = pl && a->p.layout == DAV1D_PIXEL_LAYOUT_I420;
        const int w = (a->p.w + ssx) >> ssx, h = (a->p.h + ssy) >> ssy;
        const ptrdiff_t sa = a->stride[!!pl], sb = b->stride[!!pl];
        for (int y = 0; y < h; y++)
            if (memcmp((const uint8_t *)a->data[pl] + y * sa, (const uint8_t *)b->data[pl] + y * sb, (size_t)w * bps))
                return 0;
    }
    return 1;
}

static int params(const char *path)
{
    Dec d;
    if (dec_open(&d, path, 0) < 0)
        return 1;
    Dav1dPicture p;
    int n = 0, r;
    while ((r = dec_next(&d, &p)) == 0) {
        printf("%d ", n++);
        print_grain(p.frame_hdr);
        dav1d_picture_unref(&p);
    }
    dec_close(&d);
    return r < 0;
}

static int edit(const char *orig, const char *patched)
{
    Dec a, b, c;
    if (dec_open(&a, orig, 0) < 0 || dec_open(&b, patched, 0) < 0 || dec_open(&c, patched, 1) < 0)
        return 1;
    int n = 0, changed = 0, fail = 0, ra, rb, rc;
    for (;;) {
        Dav1dPicture pa, pb, pc;
        ra = dec_next(&a, &pa);
        rb = dec_next(&b, &pb);
        rc = dec_next(&c, &pc);
        if (ra || rb || rc) {
            if (ra != rb || rb != rc) {
                fprintf(stderr, "picture %d: different number of pictures (%d %d %d)\n", n, ra, rb, rc);
                fail = 1;
            }
            if (!ra) dav1d_picture_unref(&pa);
            if (!rb) dav1d_picture_unref(&pb);
            if (!rc) dav1d_picture_unref(&pc);
            break;
        }
        /* 1. grain off: identical pictures */
        if (!same_pixels(&pa, &pb)) {
            fprintf(stderr, "picture %d: grain-off decodes differ\n", n);
            fail = 1;
        }
        /* 2. identical grain parameters except the seed */
        const Dav1dFrameHeader *ha = pa.frame_hdr, *hb = pb.frame_hdr;
        Dav1dFilmGrainData ga = ha->film_grain.data, gb = hb->film_grain.data;
        if (ha->film_grain.present != hb->film_grain.present) {
            fprintf(stderr, "picture %d: apply_grain differs\n", n);
            fail = 1;
        }
        if (ga.seed != gb.seed)
            changed++;
        ga.seed = gb.seed = 0;
        if (memcmp(&ga, &gb, sizeof(ga))) {
            fprintf(stderr, "picture %d: grain parameters differ beyond the seed\n", n);
            fail = 1;
        }
        /* 3. grain on: ORIGINAL's picture + PATCHED's grain == PATCHED decoded with grain */
        Dav1dFrameHeader fh = *ha;
        fh.film_grain = hb->film_grain;
        Dav1dPicture tmp = pa, withg;
        tmp.frame_hdr = &fh;
        memset(&withg, 0, sizeof(withg));
        if (dav1d_apply_grain(a.c, &withg, &tmp) < 0) {
            fprintf(stderr, "picture %d: dav1d_apply_grain failed\n", n);
            fail = 1;
        } else {
            if (!same_pixels(&withg, &pc)) {
                fprintf(stderr, "picture %d: grain-on decode is not the original + new-seed grain\n", n);
                fail = 1;
            }
            dav1d_picture_unref(&withg);
        }
        dav1d_picture_unref(&pa);
        dav1d_picture_unref(&pb);
        dav1d_picture_unref(&pc);
        n++;
    }
    dec_close(&a);
    dec_close(&b);
    dec_close(&c);
    if (ra < 0 || rb < 0 || rc < 0)
        fail = 1;
    printf("%d pictures, %d seeds changed: %s\n", n, changed, fail ? "FAIL" : "OK");
    return fail;
}

int main(int argc, char **argv)
{
    if (argc == 3 && !strcmp(argv[1], "params"))
        return params(argv[2]);
    if (argc == 4 && !strcmp(argv[1], "edit"))
        return edit(argv[2], argv[3]);
    fprintf(stderr, "usage: av1fg-dav1d-check params FILE | edit ORIGINAL PATCHED\n");
    return 2;
}
