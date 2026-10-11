/*
 * av1fg_score: seed badness from the templates and block offsets.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#include "av1fg_score.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const char *const av1fg_metric_names[AV1FG_NM] = {
    "L4", "L8", "L16", "V8", "V16", "CbL8", "CbV8", "CrL8", "CrV8", "ident",
};
const double av1fg_metric_weights[AV1FG_NM] = {
    0.5, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.5,
};

/* 64 seeds spanning the quality range evenly (av1fg_refseeds.inc, built by tests/mkpool.py) */
const uint16_t av1fg_ref_seeds[AV1FG_NREF] = {
#include "av1fg_refseeds.inc"
};

#define LAT 32 /* lattice positions per dimension (offsets 0..15 + window starts) */

/* per-offset sums for one plane and one window size */
typedef struct PlaneScale {
    double s1[256], s2[256], v1[256], v2[256];
    int nwin;   /* windows per crop */
    int npix;   /* pixels per window */
} PlaneScale;

typedef struct PlaneStats {
    double sum[256], sq[256];
    int npix;   /* pixels per crop */
    PlaneScale sc[3];
} PlaneStats;

struct Av1fgScratch {
    Av1fgTemplates t[AV1FG_BATCH];
    Av1fgBatch batch;
    int64_t ii[AV1FG_TPL_H + 1][AV1FG_TPL_W + 1];  /* integral image of T */
    int64_t iq[AV1FG_TPL_H + 1][AV1FG_TPL_W + 1];  /* integral image of T^2 */
    double a[4][LAT][LAT];                          /* lattice values: s, s^2, v, v^2 */
    double p[4][LAT + 4][LAT + 4];                  /* strided prefix sums */
    PlaneStats ps[3];
    uint8_t row[2][2048];                           /* block offsets of two stripes */
};

static int32_t *g_pos; /* LFSR positions, built once by av1fg_scratch_alloc() */

int av1fg_lfsr_pos(unsigned state)
{
    if (!g_pos) {
        int32_t *q = malloc(65536 * sizeof(*q));
        if (!q)
            return -1;
        q[0] = -1;
        unsigned r = 1;
        for (int i = 0; i < 65535; i++) {
            q[r] = i;
            const unsigned bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
            r = (r >> 1) | (bit << 15);
        }
        g_pos = q;
    }
    return g_pos[state & 0xffff];
}

Av1fgScratch *av1fg_scratch_alloc(void)
{
    av1fg_lfsr_pos(1);
    return calloc(1, sizeof(Av1fgScratch));
}

void av1fg_scratch_free(Av1fgScratch *s)
{
    free(s);
}

/* stats of every 256 crop offsets of one plane.
 * sx/sy: plane subsampling; T: template; windows of b x b luma pixels. */
static void plane_stats(Av1fgScratch *sc, const int16_t (*T)[AV1FG_TPL_W], int sx, int sy, int nscales,
                        const int *bs, PlaneStats *ps)
{
    const int by = sy ? 6 : 9, bx = sx ? 6 : 9;       /* crop origin at offset 0 */
    const int uy = sy ? 1 : 2, ux = sx ? 1 : 2;       /* pixels per offset step */
    const int cy = 32 >> sy, cx = 32 >> sx;           /* crop size */
    const int h = by + uy * 15 + cy, w = bx + ux * 15 + cx;
    /* integral images over [0,h) x [0,w) */
    for (int x = 0; x <= w; x++)
        sc->ii[0][x] = sc->iq[0][x] = 0;
    for (int y = 0; y < h; y++) {
        int64_t rs = 0, rq = 0;
        sc->ii[y + 1][0] = sc->iq[y + 1][0] = 0;
        for (int x = 0; x < w; x++) {
            const int v = T[y][x];
            rs += v;
            rq += (int64_t)v * v;
            sc->ii[y + 1][x + 1] = sc->ii[y][x + 1] + rs;
            sc->iq[y + 1][x + 1] = sc->iq[y][x + 1] + rq;
        }
    }
#define BOX(I, y0, x0, hh, ww) ((I)[(y0) + (hh)][(x0) + (ww)] - (I)[(y0)][(x0) + (ww)] - (I)[(y0) + (hh)][(x0)] + (I)[(y0)][(x0)])
    ps->npix = cy * cx;
    for (int o = 0; o < 256; o++) {
        const int oy = by + uy * (o & 15), ox = bx + ux * (o >> 4);
        ps->sum[o] = (double)BOX(sc->ii, oy, ox, cy, cx);
        ps->sq[o] = (double)BOX(sc->iq, oy, ox, cy, cx);
    }
    for (int k = 0; k < nscales; k++) {
        PlaneScale *s = &ps->sc[k];
        const int wy = bs[k] >> sy, wx = bs[k] >> sx;         /* window */
        const int ty = (bs[k] / 2) >> sy, tx = (bs[k] / 2) >> sx; /* stride */
        const int ly = ty / uy, lx = tx / ux;                 /* lattice step (b/4) */
        const int ny = (cy - wy) / ty + 1, nx = (cx - wx) / tx + 1;
        const int npos_y = 15 + ly * (ny - 1) + 1, npos_x = 15 + lx * (nx - 1) + 1;
        s->nwin = ny * nx;
        s->npix = wy * wx;
        const int64_t n = wy * wx;
        for (int j = 0; j < npos_y; j++)
            for (int i = 0; i < npos_x; i++) {
                const int y0 = by + uy * j, x0 = bx + ux * i;
                const int64_t bsum = BOX(sc->ii, y0, x0, wy, wx);
                const int64_t bsq = BOX(sc->iq, y0, x0, wy, wx);
                const double v = (double)(n * bsq - bsum * bsum); /* n^2 x local variance */
                sc->a[0][j][i] = (double)bsum;
                sc->a[1][j][i] = (double)(bsum * bsum);
                sc->a[2][j][i] = v;
                sc->a[3][j][i] = v * v;
            }
        /* prefix sums with stride (ly, lx): P[j][i] = A[j][i] + P[j-ly][i] + P[j][i-lx] - P[j-ly][i-lx] */
        for (int q = 0; q < 4; q++)
            for (int j = 0; j < npos_y; j++)
                for (int i = 0; i < npos_x; i++) {
                    double v = sc->a[q][j][i];
                    if (j >= ly)
                        v += sc->p[q][j - ly][i];
                    if (i >= lx)
                        v += sc->p[q][j][i - lx];
                    if (j >= ly && i >= lx)
                        v -= sc->p[q][j - ly][i - lx];
                    sc->p[q][j][i] = v;
                }
        for (int o = 0; o < 256; o++) {
            const int j0 = o & 15, i0 = o >> 4;
            const int j1 = j0 + ly * (ny - 1), i1 = i0 + lx * (nx - 1);
            double r[4];
            for (int q = 0; q < 4; q++) {
                double v = sc->p[q][j1][i1];
                if (j0 >= ly)
                    v -= sc->p[q][j0 - ly][i1];
                if (i0 >= lx)
                    v -= sc->p[q][j1][i0 - lx];
                if (j0 >= ly && i0 >= lx)
                    v += sc->p[q][j0 - ly][i0 - lx];
                r[q] = v;
            }
            s->s1[o] = r[0];
            s->s2[o] = r[1];
            s->v1[o] = r[2];
            s->v2[o] = r[3];
        }
    }
#undef BOX
}

/* frame statistics: L_b and V_b for the scales, weighted by how many blocks use each offset */
static void frame_stats(const PlaneStats *ps, const int *cnt, int nscales, double *lf, double *cv)
{
    double n = 0, s = 0, q = 0;
    for (int o = 0; o < 256; o++) {
        if (!cnt[o])
            continue;
        n += (double)cnt[o] * ps->npix;
        s += cnt[o] * ps->sum[o];
        q += cnt[o] * ps->sq[o];
    }
    const double mean = s / n, var = q / n - mean * mean;
    for (int k = 0; k < nscales; k++) {
        const PlaneScale *p = &ps->sc[k];
        double w = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        for (int o = 0; o < 256; o++) {
            if (!cnt[o])
                continue;
            w += (double)cnt[o] * p->nwin;
            a1 += cnt[o] * p->s1[o];
            a2 += cnt[o] * p->s2[o];
            b1 += cnt[o] * p->v1[o];
            b2 += cnt[o] * p->v2[o];
        }
        const double np = p->npix;
        const double m1 = a1 / w / np, m2 = a2 / w / (np * np);  /* window means */
        const double vm = m2 - m1 * m1;
        lf[k] = var > 0 ? vm / var : 0;
        const double v1 = b1 / w, v2 = b2 / w;                  /* local variances (x np^2) */
        const double vv = v2 - v1 * v1;
        cv[k] = v1 > 0 ? sqrt(vv > 0 ? vv : 0) / v1 : 0;
    }
}

static unsigned stripe_reg(unsigned seed, int n)
{
    return seed ^ (((unsigned)(n * 37 + 178) & 255) << 8) ^ ((unsigned)(n * 173 + 105) & 255);
}

int av1fg_seed_degenerate(unsigned seed, const Av1fgGrainShape *sh, int height)
{
    seed &= 0xffff;
    if (!seed)
        return 1;
    const Av1FilmGrain *g = &sh->fg;
    if (!sh->mono && (g->num_cb_points || g->chroma_scaling_from_luma) && !(seed ^ 0xb524))
        return 1;
    if (!sh->mono && (g->num_cr_points || g->chroma_scaling_from_luma) && !(seed ^ 0x49d8))
        return 1;
    const int nr = ((height + 1) / 2 + 15) / 16;
    for (int n = 0; n < nr; n++)
        if (!stripe_reg(seed, n))
            return 1;
    return 0;
}

/* stats of one seed whose templates are in t */
static void stats_from_templates(Av1fgScratch *sc, const Av1fgTemplates *t, const Av1fgGrainShape *sh, int width,
                                 int height, unsigned seed, Av1fgStats *st)
{
    const Av1FilmGrain *g = &sh->fg;
    const int luma = g->num_y_points > 0;
    const int cb = !sh->mono && (g->num_cb_points || g->chroma_scaling_from_luma);
    const int cr = !sh->mono && (g->num_cr_points || g->chroma_scaling_from_luma);
    /* block offsets (7.18.3.5): histogram and neighbouring identical offsets */
    const int rows = ((height + 1) / 2 + 15) / 16, cols = ((width + 1) / 2 + 15) / 16;
    int cnt[256] = { 0 }, ident = 0;
    for (int r = 0; r < rows; r++) {
        uint8_t *cur = sc->row[r & 1], *prev = sc->row[!(r & 1)];
        unsigned reg = stripe_reg(seed, r);
        for (int c = 0; c < cols && c < 2048; c++) {
            const unsigned bit = (reg ^ (reg >> 1) ^ (reg >> 3) ^ (reg >> 12)) & 1;
            reg = (reg >> 1) | (bit << 15);
            const uint8_t v = (uint8_t)(reg >> 8);
            cur[c] = v;
            cnt[v]++;
            if (c && cur[c - 1] == v)
                ident++;
            if (r && prev[c] == v)
                ident++;
        }
    }
    st->m[AV1FG_M_IDENT] = ident;
    static const int bs_luma[3] = { 4, 8, 16 }, bs_chroma[1] = { 8 };
    double lf[3], cv[3];
    if (luma) {
        plane_stats(sc, (const int16_t(*)[AV1FG_TPL_W])t->y, 0, 0, 3, bs_luma, &sc->ps[0]);
        frame_stats(&sc->ps[0], cnt, 3, lf, cv);
        st->m[AV1FG_M_L4] = lf[0];
        st->m[AV1FG_M_L8] = lf[1];
        st->m[AV1FG_M_L16] = lf[2];
        st->m[AV1FG_M_V8] = cv[1];
        st->m[AV1FG_M_V16] = cv[2];
    }
    for (int pl = 0; pl < 2; pl++) {
        if (!(pl ? cr : cb))
            continue;
        plane_stats(sc, (const int16_t(*)[AV1FG_TPL_W])(pl ? t->cr : t->cb), sh->ssx, sh->ssy, 1, bs_chroma,
                    &sc->ps[1 + pl]);
        frame_stats(&sc->ps[1 + pl], cnt, 1, lf, cv);
        st->m[pl ? AV1FG_M_CR_L8 : AV1FG_M_CB_L8] = lf[0];
        st->m[pl ? AV1FG_M_CR_V8 : AV1FG_M_CB_V8] = cv[0];
    }
    st->valid = 1;
}

void av1fg_seed_stats_n(Av1fgScratch *sc, const Av1fgGrainShape *sh, int width, int height, const unsigned *seeds,
                        int n, Av1fgStats *st)
{
    const Av1FilmGrain *g = &sh->fg;
    const int planes = (g->num_y_points > 0 ? 1 : 0) |
                       (!sh->mono && (g->num_cb_points || g->chroma_scaling_from_luma) ? 2 : 0) |
                       (!sh->mono && (g->num_cr_points || g->chroma_scaling_from_luma) ? 4 : 0);
    for (int i0 = 0; i0 < n; i0 += AV1FG_BATCH) {
        unsigned ok[AV1FG_BATCH];
        int idx[AV1FG_BATCH], k = 0;
        for (int i = i0; i < n && i < i0 + AV1FG_BATCH; i++) {
            memset(&st[i], 0, sizeof(st[i]));
            if (av1fg_seed_degenerate(seeds[i], sh, height))
                continue;
            ok[k] = seeds[i] & 0xffff;
            idx[k++] = i;
        }
        if (!k)
            continue;
        av1fg_templates_batch(sh, ok, k, planes, &sc->batch, sc->t);
        for (int j = 0; j < k; j++)
            stats_from_templates(sc, &sc->t[j], sh, width, height, ok[j], &st[idx[j]]);
    }
}

void av1fg_seed_stats(Av1fgScratch *sc, const Av1fgGrainShape *sh, int width, int height, unsigned seed,
                      Av1fgStats *st)
{
    av1fg_seed_stats_n(sc, sh, width, height, &seed, 1, st);
}

static void sort_d(double *v, int n)
{
    for (int i = 1; i < n; i++) {
        const double x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = x;
    }
}

void av1fg_norm_init(Av1fgScratch *sc, const Av1fgGrainShape *sh, int width, int height, int nref, Av1fgNorm *n)
{
    Av1fgStats all[AV1FG_NREF], st[AV1FG_NREF];
    unsigned seeds[AV1FG_NREF];
    if (nref < 8 || nref > AV1FG_NREF)
        nref = AV1FG_NREF;
    /* the reference seeds are ordered by quality quantile; a subset of 1/step takes one of
     * every step seeds, alternating the position within each group so that its quantiles
     * stay centred */
    const int step = AV1FG_NREF / nref;
    nref = AV1FG_NREF / step;
    for (int i = 0; i < nref; i++)
        seeds[i] = av1fg_ref_seeds[i * step + (step > 1 ? (i & 1 ? step / 2 : (step - 1) / 2) : 0)];
    av1fg_seed_stats_n(sc, sh, width, height, seeds, nref, all);
    int k = 0;
    for (int i = 0; i < nref; i++)
        if (all[i].valid)
            st[k++] = all[i];
    memset(n, 0, sizeof(*n));
    for (int m = 0; m < AV1FG_NM; m++) {
        double s = 0, q = 0;
        for (int i = 0; i < k; i++)
            s += st[i].m[m];
        const double mean = k ? s / k : 0;
        for (int i = 0; i < k; i++)
            q += (st[i].m[m] - mean) * (st[i].m[m] - mean);
        const double sd = k > 1 ? sqrt(q / (k - 1)) : 0;
        n->mean[m] = mean;
        n->inv_sd[m] = sd > 0 ? 1.0 / sd : 0;
    }
    for (int i = 0; i < k; i++)
        n->ref_b[i] = av1fg_badness(&st[i], n);
    n->nref = k;
    sort_d(n->ref_b, k);
}

double av1fg_badness(const Av1fgStats *st, const Av1fgNorm *n)
{
    if (!st->valid)
        return 1e9;
    double b = 0;
    for (int m = 0; m < AV1FG_NM; m++)
        b += av1fg_metric_weights[m] * (st->m[m] - n->mean[m]) * n->inv_sd[m];
    return b;
}

double av1fg_percentile(double b, const Av1fgNorm *n)
{
    if (!n->nref)
        return 50;
    if (b >= 1e8)
        return 100;
    if (b <= n->ref_b[0])
        return 100.0 * 0.5 / n->nref;
    if (b >= n->ref_b[n->nref - 1])
        return 100.0 * (n->nref - 0.5) / n->nref;
    int i = 0;
    while (n->ref_b[i + 1] < b)
        i++;
    const double f = (b - n->ref_b[i]) / (n->ref_b[i + 1] - n->ref_b[i]);
    return 100.0 * (i + 0.5 + f) / n->nref;
}
