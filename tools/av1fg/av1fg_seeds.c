/*
 * av1fg_seeds: rate a file's grain seeds and choose new ones.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#include "av1fg_seeds.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

const uint16_t av1fg_pool[] = {
#include "av1fg_pool.inc"
};
const int av1fg_pool_size = (int)(sizeof(av1fg_pool) / sizeof(av1fg_pool[0]));

void av1fg_seed_opts_default(Av1fgSeedOpts *o)
{
    memset(o, 0, sizeof(*o));
    o->candidates = 15; /* + the file's own seed = two batches of 8 */
    o->bad_pct = 10;
    o->only_bad = 0;
    o->overlap_window = 3;
    o->max_sets = 4096;
    o->threads = 0;
}

void av1fg_report_free(Av1fgSeedReport *r)
{
    free(r->fs);
    free(r->sets);
    memset(r, 0, sizeof(*r));
}

static double now_s(void)
{
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

/* ---------------- threads ---------------- */

typedef void (*JobFn)(void *ctx, int tid, int nthreads);
typedef struct JobArg {
    JobFn fn;
    void *ctx;
    int tid, n;
} JobArg;

#ifdef _WIN32
static DWORD WINAPI job_main(LPVOID p)
{
    JobArg *a = p;
    a->fn(a->ctx, a->tid, a->n);
    return 0;
}
#else
static void *job_main(void *p)
{
    JobArg *a = p;
    a->fn(a->ctx, a->tid, a->n);
    return NULL;
}
#endif

static int ncpu(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/* run fn(ctx, tid, n) on n threads; work is split by tid, so results do not depend on n */
static void run_jobs(JobFn fn, void *ctx, int n)
{
    if (n <= 1) {
        fn(ctx, 0, 1);
        return;
    }
    JobArg a[64];
    if (n > 64)
        n = 64;
#ifdef _WIN32
    HANDLE h[64];
#else
    pthread_t h[64];
#endif
    int started = 0;
    for (int i = 1; i < n; i++) {
        a[i].fn = fn;
        a[i].ctx = ctx;
        a[i].tid = i;
        a[i].n = n;
#ifdef _WIN32
        h[i] = CreateThread(NULL, 0, job_main, &a[i], 0, NULL);
        if (!h[i])
            break;
#else
        if (pthread_create(&h[i], NULL, job_main, &a[i]))
            break;
#endif
        started = i;
    }
    if (started < n - 1) { /* could not start every thread: do it all here after joining */
        for (int i = 1; i <= started; i++) {
#ifdef _WIN32
            WaitForSingleObject(h[i], INFINITE);
            CloseHandle(h[i]);
#else
            pthread_join(h[i], NULL);
#endif
        }
        /* the started threads did their share; the rest is done serially */
        for (int t = started + 1; t < n; t++)
            fn(ctx, t, n);
        fn(ctx, 0, n);
        return;
    }
    fn(ctx, 0, n);
    for (int i = 1; i < n; i++) {
#ifdef _WIN32
        WaitForSingleObject(h[i], INFINITE);
        CloseHandle(h[i]);
#else
        pthread_join(h[i], NULL);
#endif
    }
}

/* ---------------- parameter groups ---------------- */

static void shape_of(const Av1fgScan *s, int param, Av1fgGrainShape *sh)
{
    const Av1fgParamSet *p = &s->params[param];
    memset(sh, 0, sizeof(*sh));
    sh->bit_depth = p->bit_depth;
    sh->ssx = p->subsampling_x;
    sh->ssy = p->subsampling_y;
    sh->mono = p->mono_chrome;
    sh->fg = p->fg;
    sh->fg.apply_grain = 1;
}

/* the fields a template depends on (not the scaling functions) */
typedef struct ShapeKey {
    int16_t bit_depth, ssx, ssy, mono, has_y, has_cb, has_cr, csfl, lag, shift, gss;
    int16_t width, height;
    uint8_t ary[24], arcb[25], arcr[25];
} ShapeKey;

static void key_of(const Av1fgGrainShape *sh, int w, int h, ShapeKey *k)
{
    memset(k, 0, sizeof(*k));
    const Av1FilmGrain *g = &sh->fg;
    k->bit_depth = (int16_t)sh->bit_depth;
    k->ssx = (int16_t)sh->ssx;
    k->ssy = (int16_t)sh->ssy;
    k->mono = (int16_t)sh->mono;
    k->has_y = g->num_y_points > 0;
    k->has_cb = g->num_cb_points > 0;
    k->has_cr = g->num_cr_points > 0;
    k->csfl = g->chroma_scaling_from_luma;
    k->lag = g->ar_coeff_lag;
    k->shift = g->ar_coeff_shift_minus_6;
    k->gss = g->grain_scale_shift;
    k->width = (int16_t)w;
    k->height = (int16_t)h;
    const int npl = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
    if (k->has_y)
        memcpy(k->ary, g->ar_coeffs_y_plus_128, npl);
    if (k->has_cb || k->csfl)
        memcpy(k->arcb, g->ar_coeffs_cb_plus_128, npl + k->has_y);
    if (k->has_cr || k->csfl)
        memcpy(k->arcr, g->ar_coeffs_cr_plus_128, npl + k->has_y);
}

static uint32_t hash_key(const ShapeKey *k)
{
    const uint8_t *b = (const uint8_t *)k;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < sizeof(*k); i++)
        h = (h ^ b[i]) * 16777619u;
    return h;
}

typedef struct Work {
    Av1fgScan *s;
    const Av1fgSeedOpts *o;
    Av1fgSeedReport *r;
    int64_t *order;          /* frame indices to process, in display order */
    int64_t n;
    int *set_of;             /* per order entry */
    int64_t *frame_w;        /* per order entry: width, height */
    int64_t *frame_h;
    uint16_t *cand;          /* n x K candidate seeds */
    float *cand_b;           /* n x K badness */
    float *old_b;
    int K;
    int nref;                /* reference seeds per parameter set */
} Work;

static void job_norms(void *ctx, int tid, int nt)
{
    Work *w = ctx;
    Av1fgScratch *sc = av1fg_scratch_alloc();
    if (!sc)
        return;
    for (int i = tid; i < w->r->n_sets; i += nt) {
        Av1fgSet *st = &w->r->sets[i];
        av1fg_norm_init(sc, &st->sh, st->width, st->height, w->nref, &st->norm);
    }
    av1fg_scratch_free(sc);
}

static void job_frames(void *ctx, int tid, int nt)
{
    Work *w = ctx;
    Av1fgScratch *sc = av1fg_scratch_alloc();
    if (!sc)
        return;
    for (int64_t k = tid; k < w->n; k += nt) {
        const Av1fgFrame *fr = &w->s->frames[w->order[k]];
        const Av1fgSet *set = &w->r->sets[w->set_of[k]];
        Av1fgGrainShape sh;
        shape_of(w->s, fr->param, &sh);
        unsigned seeds[1 + 256];
        Av1fgStats st[1 + 256];
        seeds[0] = fr->seed;
        for (int j = 0; j < w->K; j++)
            seeds[1 + j] = w->cand[k * w->K + j];
        av1fg_seed_stats_n(sc, &sh, (int)w->frame_w[k], (int)w->frame_h[k], seeds, 1 + w->K, st);
        w->old_b[k] = (float)av1fg_badness(&st[0], &set->norm);
        for (int j = 0; j < w->K; j++)
            w->cand_b[k * w->K + j] = (float)av1fg_badness(&st[1 + j], &set->norm);
    }
    av1fg_scratch_free(sc);
}

static int64_t gcd64(int64_t a, int64_t b)
{
    while (b) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int circ_dist(int a, int b)
{
    int d = a - b;
    if (d < 0)
        d = -d;
    return d < 65535 - d ? d : 65535 - d;
}

/* more than half of a template shared: LFSR positions closer than half the template */
static int overlaps(unsigned a, unsigned b, int chroma_half)
{
    if (circ_dist(av1fg_lfsr_pos(a), av1fg_lfsr_pos(b)) < AV1FG_TPL_H * AV1FG_TPL_W / 2)
        return 1;
    if (circ_dist(av1fg_lfsr_pos(a ^ 0xb524), av1fg_lfsr_pos(b ^ 0xb524)) < chroma_half)
        return 1;
    if (circ_dist(av1fg_lfsr_pos(a ^ 0x49d8), av1fg_lfsr_pos(b ^ 0x49d8)) < chroma_half)
        return 1;
    return 0;
}

static int cmp_float(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

int av1fg_choose_seeds(Av1fgScan *s, const Av1fgSeedOpts *o, const uint16_t *list, size_t nlist, int write,
                       Av1fgSeedReport *r)
{
    (void)write;
    const double t0 = now_s();
    memset(r, 0, sizeof(*r));
    if (!s->n_frames)
        return 0;
    r->fs = calloc(s->n_frames, sizeof(*r->fs));
    if (!r->fs)
        return -1;
    for (size_t i = 0; i < s->n_frames; i++)
        r->fs[i].set = -1;
    Work w;
    memset(&w, 0, sizeof(w));
    w.s = s;
    w.o = o;
    w.r = r;
    w.K = list ? 0 : (o->candidates > 0 ? o->candidates : 1);
    w.order = malloc(s->n_display * sizeof(*w.order) + 8);
    w.set_of = malloc(s->n_display * sizeof(*w.set_of) + 8);
    w.frame_w = malloc(s->n_display * sizeof(*w.frame_w) + 8);
    w.frame_h = malloc(s->n_display * sizeof(*w.frame_h) + 8);
    ShapeKey *keys = malloc(s->n_display * sizeof(*keys) + 8);
    int ret = -1;
    if (!w.order || !w.set_of || !w.frame_w || !w.frame_h || !keys)
        goto out;
    /* frames to score: first display of each coded frame with a seed and visible grain */
    for (size_t d = 0; d < s->n_display; d++) {
        const int64_t fi = s->display[d];
        const Av1fgFrame *fr = &s->frames[fi];
        if (fr->display_idx != (int64_t)d || !fr->n_loc || fr->param < 0)
            continue;
        const Av1FilmGrain *g = &s->params[fr->param].fg;
        if (!g->num_y_points && !g->num_cb_points && !g->num_cr_points && !g->chroma_scaling_from_luma)
            continue;
        w.order[w.n] = fi;
        w.frame_w[w.n] = fr->upscaled_width;
        w.frame_h[w.n] = fr->height;
        Av1fgGrainShape sh;
        shape_of(s, fr->param, &sh);
        key_of(&sh, fr->upscaled_width, fr->height, &keys[w.n]);
        w.n++;
    }
    if (!w.n) {
        ret = 0;
        goto out;
    }
    /* groups: one per distinct template shape; with too many (per-frame
     * parameters), one per run of similar shapes that starts at a key frame */
    {
        int cap = 1024;
        r->sets = malloc(cap * sizeof(*r->sets));
        ShapeKey *skeys = malloc(cap * sizeof(*skeys));
        uint32_t hcap = 1;
        while (hcap < (uint32_t)w.n * 2)
            hcap <<= 1;
        int *ht = malloc(hcap * sizeof(*ht));
        if (!r->sets || !skeys || !ht) {
            free(skeys);
            free(ht);
            goto out;
        }
        for (uint32_t i = 0; i < hcap; i++)
            ht[i] = -1;
        for (int64_t k = 0; k < w.n; k++) {
            uint32_t h = hash_key(&keys[k]) & (hcap - 1);
            while (ht[h] >= 0 && memcmp(&skeys[ht[h]], &keys[k], sizeof(ShapeKey)))
                h = (h + 1) & (hcap - 1);
            if (ht[h] < 0) {
                if (r->n_sets == cap) {
                    cap *= 2;
                    Av1fgSet *ns = realloc(r->sets, cap * sizeof(*ns));
                    ShapeKey *nk = realloc(skeys, cap * sizeof(*nk));
                    if (!ns || !nk) {
                        free(ns ? ns : r->sets);
                        r->sets = NULL;
                        free(nk ? nk : skeys);
                        free(ht);
                        goto out;
                    }
                    r->sets = ns;
                    skeys = nk;
                }
                ht[h] = r->n_sets;
                skeys[r->n_sets] = keys[k];
                Av1fgSet *st = &r->sets[r->n_sets++];
                memset(st, 0, sizeof(*st));
                shape_of(s, s->frames[w.order[k]].param, &st->sh);
                st->width = (int)w.frame_w[k];
                st->height = (int)w.frame_h[k];
                st->first_display = s->frames[w.order[k]].display_idx;
            }
            w.set_of[k] = ht[h];
        }
        free(skeys);
        free(ht);
    }
    for (int64_t k = 0; k < w.n; k++) {
        Av1fgSet *st = &r->sets[w.set_of[k]];
        st->frames++;
        st->last_display = s->frames[w.order[k]].display_idx;
        r->fs[w.order[k]].set = w.set_of[k];
    }
    const int nt = o->threads > 0 ? o->threads : (ncpu() < 8 ? ncpu() : 8);
    /* per-frame grain parameters (e.g. SVT-AV1 through ffmpeg): fewer reference seeds per set */
    w.nref = o->nref ? o->nref : r->n_sets > o->max_sets ? AV1FG_NREF / 2 : AV1FG_NREF;
    r->nref = w.nref;
    run_jobs(job_norms, &w, nt);
    /* candidates: frame k takes pool seeds k*K .. k*K+K-1 (mod pool) */
    w.old_b = malloc(w.n * sizeof(*w.old_b));
    if (w.K) {
        w.cand = malloc((size_t)w.n * w.K * sizeof(*w.cand));
        w.cand_b = malloc((size_t)w.n * w.K * sizeof(*w.cand_b));
        if (!w.cand || !w.cand_b)
            goto out;
        /* the pool is sorted by LFSR position; a stride of about 0.618 x its size (odd, so the walk
         * visits every entry once per cycle) takes nearby frames' candidates from far-apart templates */
        int64_t stride = (int64_t)(av1fg_pool_size * 0.6180339887) | 1;
        while (gcd64(stride, av1fg_pool_size) != 1)
            stride += 2;
        for (int64_t k = 0; k < w.n; k++)
            for (int j = 0; j < w.K; j++)
                w.cand[k * w.K + j] = av1fg_pool[((k * w.K + j) % av1fg_pool_size) * stride % av1fg_pool_size];
    }
    if (!w.old_b)
        goto out;
    run_jobs(job_frames, &w, nt);
    /* sequential choice */
    Av1fgScratch *sc = av1fg_scratch_alloc();
    if (!sc)
        goto out;
    for (int64_t k = 0; k < w.n; k++) {
        Av1fgFrame *fr = &s->frames[w.order[k]];
        Av1fgFrameScore *fs = &r->fs[w.order[k]];
        const Av1fgSet *set = &r->sets[w.set_of[k]];
        fs->b_old = w.old_b[k];
        fs->p_old = (float)av1fg_percentile(w.old_b[k], &set->norm);
        r->n_scored++;
        if (fs->p_old > 100 - o->bad_pct)
            r->n_bad_old++;
        if (fs->p_old > 98)
            r->n_vbad_old++;
        unsigned best = fr->seed;
        double best_b = w.old_b[k];
        if (list) {
            best = list[k % nlist];
            Av1fgGrainShape sh;
            shape_of(s, fr->param, &sh);
            Av1fgStats st;
            av1fg_seed_stats(sc, &sh, (int)w.frame_w[k], (int)w.frame_h[k], best, &st);
            best_b = av1fg_badness(&st, &set->norm);
        } else if (!o->only_bad || fs->p_old > 100 - o->bad_pct) {
            const int chalf = (set->sh.ssx ? 44 : 82) * (set->sh.ssy ? 38 : 73) / 2;
            int found = 0;
            for (int j = 0; j < w.K; j++) {
                const unsigned c = w.cand[k * w.K + j];
                const double b = w.cand_b[k * w.K + j];
                if (b >= 1e8)
                    continue;
                int clash = 0;
                for (int q = 1; q <= o->overlap_window && q <= k && !clash; q++)
                    clash = overlaps(c, s->frames[w.order[k - q]].new_seed, chalf);
                if (clash)
                    continue;
                if (!found || b < best_b) {
                    best = c;
                    best_b = b;
                    found = 1;
                }
            }
            if (!found) { /* every candidate clashed: take the best one anyway */
                r->n_fallback++;
                for (int j = 0; j < w.K; j++) {
                    const double b = w.cand_b[k * w.K + j];
                    if (b < 1e8 && (!found || b < best_b)) {
                        best = w.cand[k * w.K + j];
                        best_b = b;
                        found = 1;
                    }
                }
            }
            /* keep the file's seed when it is at least as good as every candidate */
            if (!found || best_b >= w.old_b[k]) {
                best = fr->seed;
                best_b = w.old_b[k];
            }
        }
        fr->new_seed = (uint16_t)best;
        fs->b_new = (float)best_b;
        fs->p_new = (float)av1fg_percentile(best_b, &set->norm);
        if (fr->new_seed != fr->seed)
            r->n_changed++;
        if (fs->p_new > 100 - o->bad_pct)
            r->n_bad_new++;
        if (fs->p_new > 98)
            r->n_vbad_new++;
    }
    av1fg_scratch_free(sc);
    r->secs = now_s() - t0;
    ret = 0;
out:
    free(w.order);
    free(w.set_of);
    free(w.frame_w);
    free(w.frame_h);
    free(w.cand);
    free(w.cand_b);
    free(w.old_b);
    free(keys);
    return ret;
}

static double quantile(float *v, int64_t n, double q)
{
    if (!n)
        return 0;
    int64_t i = (int64_t)(q * (n - 1) + 0.5);
    return v[i < 0 ? 0 : i >= n ? n - 1 : i];
}

void av1fg_print_report(FILE *out, const Av1fgScan *s, const Av1fgSeedOpts *o, const Av1fgSeedReport *r, int write)
{
    if (!r->n_scored) {
        fprintf(out, "no frames with visible grain to rate\n");
        return;
    }
    float *po = malloc(r->n_scored * sizeof(float)), *pn = malloc(r->n_scored * sizeof(float));
    if (!po || !pn) {
        free(po);
        free(pn);
        return;
    }
    int64_t k = 0;
    for (size_t i = 0; i < s->n_frames; i++)
        if (r->fs[i].set >= 0) {
            po[k] = r->fs[i].p_old;
            pn[k] = r->fs[i].p_new;
            k++;
        }
    qsort(po, k, sizeof(float), cmp_float);
    qsort(pn, k, sizeof(float), cmp_float);
    fprintf(out, "\ngrain seeds: %lld frames with grain, %d grain parameter sets (%d reference seeds each)\n",
            (long long)r->n_scored, r->n_sets, r->nref);
    fprintf(out, "seed percentile among random seeds under the same parameters (0 = best, 100 = worst):\n");
    fprintf(out, "                    median    p90  worst   bad (>%g)  very bad (>98)\n", 100 - o->bad_pct);
    fprintf(out, "  seeds in the file  %5.1f  %5.1f  %5.1f   %6.1f %%    %6.1f %%\n", quantile(po, k, 0.5),
            quantile(po, k, 0.9), po[k - 1], 100.0 * r->n_bad_old / k, 100.0 * r->n_vbad_old / k);
    fprintf(out, "  %s       %5.1f  %5.1f  %5.1f   %6.1f %%    %6.1f %%   (%lld seeds changed)\n",
            write ? "new seeds  " : "would become", quantile(pn, k, 0.5), quantile(pn, k, 0.9), pn[k - 1],
            100.0 * r->n_bad_new / k, 100.0 * r->n_vbad_new / k, (long long)r->n_changed);
    if (r->n_fallback)
        fprintf(out, "  (%lld frames: every candidate overlapped a recent template; best taken anyway)\n",
                (long long)r->n_fallback);
    /* worst frames */
    int64_t worst[10];
    int nw = 0;
    for (size_t i = 0; i < s->n_frames; i++) {
        if (r->fs[i].set < 0)
            continue;
        int pos = nw < 10 ? nw++ : 10;
        if (pos == 10) {
            if (r->fs[i].b_old <= r->fs[worst[9]].b_old)
                continue;
            pos = 9;
        }
        while (pos > 0 && r->fs[worst[pos - 1]].b_old < r->fs[i].b_old) {
            worst[pos] = worst[pos - 1];
            pos--;
        }
        worst[pos] = (int64_t)i;
    }
    fprintf(out, "worst seeds in the file:\n");
    for (int i = 0; i < nw; i++) {
        const Av1fgFrame *fr = &s->frames[worst[i]];
        const Av1fgFrameScore *fs = &r->fs[worst[i]];
        char ob[64];
        if (fs->b_old >= 1e8f)
            snprintf(ob, sizeof(ob), "degenerate (a stripe of identical blocks or a flat template)");
        else
            snprintf(ob, sizeof(ob), "percentile %5.1f, badness %+6.2f", fs->p_old, fs->b_old);
        fprintf(out, "  frame %7lld (decode %7lld) seed %5u: %s -> seed %5u: %5.1f, %+6.2f\n", (long long)fr->display_idx,
                (long long)fr->decode_idx, fr->seed, ob, fr->new_seed, fs->p_new, fs->b_new);
    }
    /* per parameter set */
    const int show_all = r->n_sets <= 24;
    fprintf(out, "%s:\n", show_all ? "per parameter set" : "parameter sets with the most bad seeds");
    fprintf(out, "   set  frames  display range        lag shift  median  worst   bad    -> median  worst   bad\n");
    int *idx = malloc(r->n_sets * sizeof(int));
    int64_t *nbad = calloc(r->n_sets, sizeof(int64_t));
    float **lists = calloc(r->n_sets, sizeof(float *));
    int64_t *fill = calloc(r->n_sets, sizeof(int64_t));
    if (idx && nbad && lists && fill) {
        for (int i = 0; i < r->n_sets; i++) {
            idx[i] = i;
            lists[i] = malloc(2 * r->sets[i].frames * sizeof(float) + 8);
        }
        for (size_t i = 0; i < s->n_frames; i++) {
            const int st = r->fs[i].set;
            if (st < 0 || !lists[st])
                continue;
            lists[st][fill[st]] = r->fs[i].p_old;
            lists[st][r->sets[st].frames + fill[st]] = r->fs[i].p_new;
            fill[st]++;
            if (r->fs[i].p_old > 100 - o->bad_pct)
                nbad[st]++;
        }
        if (!show_all) { /* sort by bad count, then by index */
            for (int i = 1; i < r->n_sets; i++) {
                const int x = idx[i];
                int j = i - 1;
                while (j >= 0 && nbad[idx[j]] < nbad[x]) {
                    idx[j + 1] = idx[j];
                    j--;
                }
                idx[j + 1] = x;
            }
        }
        const int nshow = show_all ? r->n_sets : 12;
        for (int q = 0; q < nshow && q < r->n_sets; q++) {
            const int i = idx[q];
            const Av1fgSet *st = &r->sets[i];
            if (!lists[i] || !fill[i])
                continue;
            float *a = lists[i], *b = lists[i] + st->frames;
            qsort(a, fill[i], sizeof(float), cmp_float);
            qsort(b, fill[i], sizeof(float), cmp_float);
            int64_t nb_new = 0;
            for (int64_t j = 0; j < fill[i]; j++)
                nb_new += b[j] > 100 - o->bad_pct;
            fprintf(out, "  %4d %7d  %8lld-%-8lld  %5d %5d   %5.1f  %5.1f %5lld    -> %5.1f  %5.1f %5lld\n", i,
                    st->frames, (long long)st->first_display, (long long)st->last_display, st->sh.fg.ar_coeff_lag,
                    st->sh.fg.ar_coeff_shift_minus_6 + 6, quantile(a, fill[i], 0.5), a[fill[i] - 1],
                    (long long)nbad[i], quantile(b, fill[i], 0.5), b[fill[i] - 1], (long long)nb_new);
        }
    }
    if (lists)
        for (int i = 0; i < r->n_sets; i++)
            free(lists[i]);
    free(lists);
    free(fill);
    free(idx);
    free(nbad);
    free(po);
    free(pn);
}

/* ---------------- --rank-all ---------------- */

typedef struct RankWork {
    const Av1fgSet *sets;
    int nsets;
    double *b; /* nsets x 65536 */
} RankWork;

static void job_rank(void *ctx, int tid, int nt)
{
    RankWork *w = ctx;
    Av1fgScratch *sc = av1fg_scratch_alloc();
    Av1fgStats *xs = malloc(1024 * sizeof(*xs));
    if (!sc || !xs) {
        av1fg_scratch_free(sc);
        free(xs);
        return;
    }
    for (int64_t job = tid; job < (int64_t)w->nsets * 64; job += nt) {
        const int si = (int)(job / 64), part = (int)(job % 64);
        const Av1fgSet *st = &w->sets[si];
        unsigned seeds[1024];
        for (unsigned i = 0; i < 1024; i++)
            seeds[i] = part * 1024u + i;
        av1fg_seed_stats_n(sc, &st->sh, st->width, st->height, seeds, 1024, xs);
        for (unsigned i = 0; i < 1024; i++)
            w->b[(size_t)si * 65536 + seeds[i]] = av1fg_badness(&xs[i], &st->norm);
    }
    free(xs);
    av1fg_scratch_free(sc);
}

int av1fg_rank_all(const Av1fgScan *s, int max_sets, FILE *out)
{
    Av1fgSeedOpts o;
    av1fg_seed_opts_default(&o);
    Av1fgSeedReport r;
    Av1fgScan *ms = (Av1fgScan *)s;
    for (size_t i = 0; i < s->n_frames; i++)
        ms->frames[i].new_seed = s->frames[i].seed;
    const uint16_t one = 1;
    if (av1fg_choose_seeds(ms, &o, &one, 1, 0, &r) < 0)
        return -1;
    const int n = r.n_sets < max_sets ? r.n_sets : max_sets;
    Av1fgSet *pick = malloc(n * sizeof(*pick));
    double *b = malloc((size_t)n * 65536 * sizeof(double));
    double *acc = calloc(65536, sizeof(double));
    if (!pick || !b || !acc) {
        free(pick);
        free(b);
        free(acc);
        av1fg_report_free(&r);
        return -1;
    }
    for (int i = 0; i < n; i++)
        pick[i] = r.sets[(int64_t)i * r.n_sets / n];
    RankWork w = { pick, n, b };
    run_jobs(job_rank, &w, ncpu() < 8 ? ncpu() : 8);
    for (int i = 0; i < n; i++) { /* z-score per set over the valid seeds */
        const double *v = b + (size_t)i * 65536;
        double sum = 0, sq = 0;
        int cnt = 0;
        for (int sd = 1; sd < 65536; sd++)
            if (v[sd] < 1e8) {
                sum += v[sd];
                cnt++;
            }
        const double mean = sum / cnt;
        for (int sd = 1; sd < 65536; sd++)
            if (v[sd] < 1e8)
                sq += (v[sd] - mean) * (v[sd] - mean);
        const double sdv = sqrt(sq / cnt);
        for (int sd = 0; sd < 65536; sd++)
            acc[sd] += v[sd] < 1e8 ? (v[sd] - mean) / sdv / n : 1e9;
    }
    for (int sd = 1; sd < 65536; sd++)
        fprintf(out, "%d %.4f\n", sd, acc[sd]);
    free(pick);
    free(b);
    free(acc);
    av1fg_report_free(&r);
    return 0;
}
