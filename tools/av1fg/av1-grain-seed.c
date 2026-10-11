/*
 * av1-grain-seed: replace unlucky AV1 film grain seeds in existing files.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#if !defined(_WIN32)
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#elif !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* fseeko */
#endif
#endif
#include "av1fg_scan.h"
#include "av1fg_seeds.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define fseek64 _fseeki64
#else
#define fseek64 fseeko
#endif
#if !defined(_WIN32)
#include <sys/stat.h>
#endif

#define VERSION "1.0"

static void usage(FILE *o)
{
    fprintf(o,
            "av1-grain-seed " VERSION ": replace unlucky AV1 film grain seeds, without re-encoding\n"
            "\n"
            "usage: av1-grain-seed [options] INPUT\n"
            "\n"
            "  (no action)       same as --analyze\n"
            "  --analyze         rate the file's grain seeds per parameter set; writes nothing\n"
            "  -o, --output OUT  write a copy of INPUT with new seeds (same size, every other\n"
            "                    byte unchanged: audio, subtitles, chapters, container)\n"
            "  --in-place        patch INPUT itself (only after a full parse succeeded)\n"
            "  --only-bad        replace only the seeds rated bad (default: every grain frame\n"
            "                    gets the best of its candidates from the pool of good seeds)\n"
            "  --candidates K    pool seeds tried per frame (default 15)\n"
            "  --bad PCT         a seed is bad when it is worse than (100 - PCT) %% of random\n"
            "                    seeds under the same grain parameters (default 10)\n"
            "  --threads N       scoring threads (default: CPU count, at most 8)\n"
            "  --set-seeds LIST  write these seeds (comma-separated, in display order,\n"
            "                    repeated), for testing\n"
            "  --dump            print every frame header: seed, offsets, grain parameters\n"
            "  --dump-display    print the grain parameters per displayed frame\n"
            "  --scores FILE     write per-frame seed ratings (old and new) as TSV\n"
            "  --track N         MP4 track_ID / Matroska track number (default: first AV1 track)\n"
            "  --strict          refuse non-conformant streams (e.g. frame header copies\n"
            "                    that differ)\n"
            "  --no-verify       do not re-parse the written file\n"
            "  -q, --quiet       errors only\n"
            "  -h, --help        this text\n"
            "\n"
            "Formats: IVF, AV1 OBU streams (Section 5 and Annex B), MP4/MOV (also\n"
            "fragmented), Matroska/WebM. Only the 16-bit grain_seed field of each\n"
            "frame header changes, so every decoder (players, TVs, hardware) shows the\n"
            "new grain. Exit status: 0 ok, 1 error, 2 usage.\n");
}

static int quiet;

static void msg(const char *fmt, ...)
{
    if (quiet)
        return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

static double now_s(void)
{
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

static const char *ftype_name(int t)
{
    static const char *n[4] = { "KEY", "INTER", "INTRA_ONLY", "SWITCH" };
    return t >= 0 && t < 4 ? n[t] : "?";
}

typedef struct DumpCtx {
    FILE *out;
} DumpCtx;

static void dump_cb(void *ctx, const Av1Obu *obu, int64_t off, const Av1fgScan *s)
{
    DumpCtx *d = ctx;
    if (!obu->has_frame_hdr)
        return;
    const Av1FrameHdr *h = &obu->frame_hdr;
    char g[1024];
    av1fg_format_grain(g, sizeof(g), &h->fg, 1);
    fprintf(d->out, "obu %lld type %d frame %lld%s%s %s show %d showable %d %dx%d oh %d refresh %02x",
            (long long)off, obu->type, (long long)(s->n_frames ? (int64_t)s->n_frames - 1 : -1),
            h->is_copy ? " copy" : "", h->show_existing_frame ? " show_existing" : "", ftype_name(h->frame_type),
            h->show_frame, h->showable_frame, h->upscaled_width, h->frame_height, h->order_hint,
            h->refresh_frame_flags);
    if (h->show_existing_frame)
        fprintf(d->out, " slot %d", h->frame_to_show_map_idx);
    if (h->seed_bit_offset >= 0)
        fprintf(d->out, " seed_bit %lld fg_bits %lld..%lld update %d",
                (long long)(obu->header_len * 8 + h->seed_bit_offset),
                (long long)(obu->header_len * 8 + h->fg_bit_offset), (long long)(obu->header_len * 8 + h->fg_bit_end),
                h->update_grain_coded);
    else if (h->fg_bit_offset >= 0)
        fprintf(d->out, " fg_bits %lld..%lld", (long long)(obu->header_len * 8 + h->fg_bit_offset),
                (long long)(obu->header_len * 8 + h->fg_bit_end));
    fprintf(d->out, " | %s\n", g);
}

/* ---------------- writing ---------------- */

/* 1 if both paths name the same existing file */
static int same_file(const char *a, const char *b)
{
    if (!strcmp(a, b))
        return 1;
#if !defined(_WIN32)
    struct stat sa, sb;
    if (!stat(a, &sa) && !stat(b, &sb))
        return sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
#endif
    return 0;
}

static int cmp_loc(const void *a, const void *b)
{
    const Av1fgLoc *x = a, *y = b;
    return x->byte < y->byte ? -1 : x->byte > y->byte;
}

/* the byte edits for the new seeds (and the Matroska CRC-32s), sorted by offset */
static int build_edits(FILE *in, const Av1fgScan *s, Av1fgEdit **out, size_t *n_out, int64_t *fields, int *crcs)
{
    Av1fgLoc *l = malloc((s->n_locs + 1) * sizeof(*l));
    size_t cap = s->n_locs * 3 + 16, n = 0, nl = 0;
    Av1fgEdit *e = malloc(cap * sizeof(*e));
    if (!l || !e) {
        free(l);
        free(e);
        return -1;
    }
    for (size_t i = 0; i < s->n_locs; i++) {
        const Av1fgFrame *fr = &s->frames[s->locs[i].frame];
        if (fr->new_seed != fr->seed)
            l[nl++] = s->locs[i];
    }
    qsort(l, nl, sizeof(*l), cmp_loc);
    for (size_t i = 0; i < nl; i++) {
        const Av1fgFrame *fr = &s->frames[l[i].frame];
        uint8_t b[3], o[3];
        const size_t nb = l[i].bit ? 3 : 2;
        if (av1fg_read_at(in, l[i].byte, b, nb) != nb)
            goto fail;
        if (av1_get_bits16(b, l[i].bit) != fr->seed) {
            fprintf(stderr, "error: seed at %lld is %u, expected %u\n", (long long)l[i].byte,
                    av1_get_bits16(b, l[i].bit), fr->seed);
            goto fail;
        }
        memcpy(o, b, nb);
        av1_put_bits16(b, l[i].bit, fr->new_seed);
        for (size_t k = 0; k < nb; k++)
            if (b[k] != o[k])
                e[n++] = (Av1fgEdit){ l[i].byte + (int64_t)k, o[k], b[k] };
    }
    *fields = (int64_t)nl;
    free(l);
    *crcs = av1fg_crc_edits(in, &s->cont, &e, &n, &cap);
    if (*crcs < 0) {
        free(e);
        return -1;
    }
    *out = e;
    *n_out = n;
    return 0;
fail:
    free(l);
    free(e);
    return -1;
}

/* -o: one sequential pass, edits applied on the way */
static int write_copy(FILE *in, const char *out, const Av1fgEdit *e, size_t n)
{
    FILE *o = fopen(out, "wb");
    if (!o)
        return -1;
    const size_t bs = 1 << 22;
    uint8_t *buf = malloc(bs);
    int ret = buf ? 0 : -1;
    int64_t off = 0;
    size_t j = 0;
    if (!ret && fseek64(in, 0, SEEK_SET))
        ret = -1;
    while (!ret) {
        const size_t got = fread(buf, 1, bs, in);
        for (; j < n && e[j].off < off + (int64_t)got; j++) {
            uint8_t *p = buf + (e[j].off - off);
            if (*p != e[j].old) {
                ret = -1;
                break;
            }
            *p = e[j].val;
        }
        if (got && fwrite(buf, 1, got, o) != got)
            ret = -1;
        off += (int64_t)got;
        if (got < bs) {
            if (ferror(in))
                ret = -1;
            break;
        }
    }
    if (!ret && j != n)
        ret = -1;
    free(buf);
    if (fclose(o))
        ret = -1;
    return ret;
}

/* --in-place: write the runs of edited bytes */
static int write_in_place(FILE *f, const Av1fgEdit *e, size_t n)
{
    uint8_t run[64];
    for (size_t i = 0; i < n;) {
        size_t k = 0;
        const int64_t start = e[i].off;
        while (i < n && k < sizeof(run) && e[i].off == start + (int64_t)k)
            run[k++] = e[i++].val;
        if (fseek64(f, start, SEEK_SET) || fwrite(run, 1, k, f) != k)
            return -1;
    }
    return fflush(f) ? -1 : 0;
}

/* re-parse the written file: same structure, the planned seeds, the same
 * parameters otherwise */
static int verify(const char *path, int track, const Av1fgScan *s)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    Av1fgScan v;
    const int r = av1fg_scan(f, track, 0, &v, NULL, NULL);
    fclose(f);
    if (r < 0) {
        fprintf(stderr, "verify: %s\n", v.err);
        av1fg_scan_free(&v);
        return -1;
    }
    int bad = 0;
    if (v.n_frames != s->n_frames || v.n_locs != s->n_locs || v.n_display != s->n_display) {
        fprintf(stderr, "verify: structure differs (%zu/%zu frames, %zu/%zu seed fields)\n", v.n_frames, s->n_frames,
                v.n_locs, s->n_locs);
        bad = 1;
    }
    for (size_t i = 0; !bad && i < s->n_frames; i++) {
        const Av1fgFrame *a = &s->frames[i], *b = &v.frames[i];
        if (b->seed != a->new_seed || b->apply_grain != a->apply_grain || b->frame_type != a->frame_type ||
            b->show_frame != a->show_frame || b->display_idx != a->display_idx ||
            (a->param >= 0) != (b->param >= 0) ||
            (a->param >= 0 && memcmp(&s->params[a->param].fg, &v.params[b->param].fg, sizeof(Av1FilmGrain)))) {
            fprintf(stderr, "verify: frame %zu differs (seed %u, expected %u)\n", i, b->seed, a->new_seed);
            bad = 1;
        }
    }
    av1fg_scan_free(&v);
    return bad ? -1 : 0;
}

static int parse_seed_list(const char *str, uint16_t **out, size_t *n)
{
    size_t cap = 16, k = 0;
    uint16_t *v = malloc(cap * sizeof(*v));
    const char *p = str;
    while (v && *p) {
        char *e;
        const long x = strtol(p, &e, 0);
        if (e == p || x < 0 || x > 65535) {
            free(v);
            return -1;
        }
        if (k == cap) {
            uint16_t *nv = realloc(v, (cap *= 2) * sizeof(*v));
            if (!nv) {
                free(v);
                return -1;
            }
            v = nv;
        }
        v[k++] = (uint16_t)x;
        p = *e == ',' ? e + 1 : e;
        if (*e && *e != ',') {
            free(v);
            return -1;
        }
    }
    if (!v || !k) {
        free(v);
        return -1;
    }
    *out = v;
    *n = k;
    return 0;
}

int main(int argc, char **argv)
{
    const char *in = NULL, *out = NULL, *set_seeds = NULL, *rank_all = NULL, *scores = NULL;
    int in_place = 0, analyze = 0, dump = 0, dump_display = 0, track = 0, strict = 0, do_verify = 1;
    Av1fgSeedOpts so;
    av1fg_seed_opts_default(&so);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT()                                                                    \
    (i + 1 < argc ? argv[++i] : (fprintf(stderr, "%s needs a value\n", a), exit(2), (char *)NULL))
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "-o") || !strcmp(a, "--output")) {
            out = NEXT();
        } else if (!strcmp(a, "--in-place")) {
            in_place = 1;
        } else if (!strcmp(a, "--analyze")) {
            analyze = 1;
        } else if (!strcmp(a, "--only-bad")) {
            so.only_bad = 1;
        } else if (!strcmp(a, "--candidates")) {
            so.candidates = atoi(NEXT());
        } else if (!strcmp(a, "--threads")) {
            so.threads = atoi(NEXT());
        } else if (!strcmp(a, "--scores")) {
            scores = NEXT();
        } else if (!strcmp(a, "--rank-all")) {
            rank_all = NEXT();
        } else if (!strcmp(a, "--bad")) {
            so.bad_pct = atof(NEXT());
        } else if (!strcmp(a, "--set-seeds")) {
            set_seeds = NEXT();
        } else if (!strcmp(a, "--dump")) {
            dump = 1;
        } else if (!strcmp(a, "--dump-display")) {
            dump_display = 1;
        } else if (!strcmp(a, "--track")) {
            track = atoi(NEXT());
        } else if (!strcmp(a, "--strict")) {
            strict = 1;
        } else if (!strcmp(a, "--no-verify")) {
            do_verify = 0;
        } else if (!strcmp(a, "-q") || !strcmp(a, "--quiet")) {
            quiet = 1;
        } else if (a[0] == '-' && a[1]) {
            fprintf(stderr, "unknown option %s\n", a);
            usage(stderr);
            return 2;
        } else if (!in) {
            in = a;
        } else {
            fprintf(stderr, "more than one input\n");
            return 2;
        }
#undef NEXT
    }
    if (!in) {
        usage(stderr);
        return 2;
    }
    if (out && in_place) {
        fprintf(stderr, "use either -o or --in-place\n");
        return 2;
    }
    if (out && same_file(in, out)) {
        fprintf(stderr, "-o names the input file; use --in-place to patch it\n");
        return 2;
    }
    if (so.candidates < 1 || so.candidates > 256 || so.bad_pct <= 0 || so.bad_pct >= 100 || so.threads < 0) {
        fprintf(stderr, "--candidates / --bad / --threads out of range\n");
        return 2;
    }
    const int write = out || in_place;
    if (!write && !dump && !dump_display && !scores)
        analyze = 1;

    const double t0 = now_s();
    FILE *f = fopen(in, "rb");
    if (!f) {
        fprintf(stderr, "error: %s: %s\n", in, strerror(errno));
        return 1;
    }
    Av1fgScan s;
    DumpCtx dc = { stdout };
    const int r = av1fg_scan(f, track, strict, &s, dump ? dump_cb : NULL, &dc);
    fclose(f);
    if (r < 0) {
        fprintf(stderr, "error: %s: %s\n", in, s.err);
        av1fg_scan_free(&s);
        return 1;
    }
    const double t_parse = now_s() - t0;
    msg("%s: %s%s%s, %dx%d %d-bit, %zu samples, %zu frames (%zu shown), %lld frame headers "
        "(%lld copies), %zu seed fields, %zu grain parameter sets; parsed in %.2f s\n",
        in, av1fg_format_name(s.cont.fmt), s.cont.info[0] ? ", " : "", s.cont.info,
        s.seq.max_frame_width_minus_1 + 1, s.seq.max_frame_height_minus_1 + 1, s.seq.bit_depth, s.cont.n_units,
        s.n_frames, s.n_display, (long long)s.n_headers, (long long)s.n_copies, s.n_locs, s.n_params, t_parse);
    if (s.warnings)
        msg("warning: %d non-conformance warnings (frame header copies that differ); --strict refuses them\n",
            s.warnings);
    if (s.n_dropped)
        msg("note: %lld OBUs outside operating point 0 were skipped (not patched)\n", (long long)s.n_dropped);

    if (dump_display) {
        for (size_t i = 0; i < s.n_display; i++) {
            const Av1fgFrame *fr = &s.frames[s.display[i]];
            char g[1024];
            Av1FilmGrain fg;
            memset(&fg, 0, sizeof(fg));
            if (fr->param >= 0) {
                fg = s.params[fr->param].fg;
                fg.apply_grain = 1;
                fg.grain_seed = fr->seed;
            }
            av1fg_format_grain(g, sizeof(g), &fg, 1);
            printf("%zu %s\n", i, g);
        }
    }
    if (!s.seq.film_grain_params_present || !s.n_locs) {
        msg("%s: no film grain in this stream; nothing to do\n", in);
        av1fg_scan_free(&s);
        return 0;
    }

    if (rank_all) {
        FILE *ro = fopen(rank_all, "w");
        const int rr = ro ? av1fg_rank_all(&s, 16, ro) : -1;
        if (ro)
            fclose(ro);
        av1fg_scan_free(&s);
        return rr < 0;
    }
    int ret = 0;
    Av1fgSeedReport rep;
    memset(&rep, 0, sizeof(rep));
    if (analyze || write || scores) {
        uint16_t *list = NULL;
        size_t nlist = 0;
        if (set_seeds && parse_seed_list(set_seeds, &list, &nlist) < 0) {
            fprintf(stderr, "--set-seeds: bad list\n");
            av1fg_scan_free(&s);
            return 2;
        }
        const double t1 = now_s();
        if (av1fg_choose_seeds(&s, &so, list, nlist, write, &rep) < 0) {
            fprintf(stderr, "error: seed selection failed\n");
            free(list);
            av1fg_scan_free(&s);
            return 1;
        }
        free(list);
        if (scores) {
            FILE *so_f = fopen(scores, "w");
            if (!so_f) {
                fprintf(stderr, "error: %s: %s\n", scores, strerror(errno));
            } else {
                fprintf(so_f, "display\tdecode\tseed\tpercentile\tbadness\tnew_seed\tnew_percentile\tnew_badness\n");
                for (size_t d = 0; d < s.n_display; d++) {
                    const Av1fgFrame *fr = &s.frames[s.display[d]];
                    const Av1fgFrameScore *fs = &rep.fs[s.display[d]];
                    if (fr->display_idx != (int64_t)d || fs->set < 0)
                        continue;
                    fprintf(so_f, "%zu\t%lld\t%u\t%.1f\t%.3f\t%u\t%.1f\t%.3f\n", d, (long long)fr->decode_idx, fr->seed,
                            fs->p_old, fs->b_old >= 1e8f ? 999.0 : fs->b_old, fr->new_seed, fs->p_new, fs->b_new);
                }
                fclose(so_f);
            }
        }
        if (!quiet || analyze)
            av1fg_print_report(analyze ? stdout : stderr, &s, &so, &rep, write);
        msg("seed scoring: %.2f s\n", now_s() - t1);
        av1fg_report_free(&rep);
    }
    if (write) {
        const char *target = in_place ? in : out;
        const double t2 = now_s();
        FILE *fi = fopen(in, in_place ? "r+b" : "rb");
        if (!fi) {
            fprintf(stderr, "error: %s: %s\n", in, strerror(errno));
            av1fg_scan_free(&s);
            return 1;
        }
        Av1fgEdit *edits = NULL;
        size_t nedits = 0;
        int64_t nw = 0;
        int ncrc = 0;
        int wr = build_edits(fi, &s, &edits, &nedits, &nw, &ncrc);
        if (!wr)
            wr = in_place ? write_in_place(fi, edits, nedits) : write_copy(fi, out, edits, nedits);
        free(edits);
        if (fclose(fi) || wr < 0) {
            fprintf(stderr, "error: writing %s failed%s\n", target, in_place ? " (the input may be partly patched)" : "");
            av1fg_scan_free(&s);
            return 1;
        }
        msg("written in %.2f s\n", now_s() - t2);
        msg("%s: wrote %lld seed fields (%zu bytes)", target, (long long)nw, nedits);
        if (ncrc)
            msg(", updated %d CRC-32 elements", ncrc);
        msg("\n");
        if (do_verify) {
            if (verify(target, track, &s) < 0) {
                fprintf(stderr, "error: verification of %s failed\n", target);
                ret = 1;
            } else {
                msg("verified: %s parses to the same frames with the new seeds\n", target);
            }
        }
    }
    msg("total %.2f s\n", now_s() - t0);
    av1fg_scan_free(&s);
    return ret;
}
