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
#define _GNU_SOURCE /* fseeko, copy_file_range */
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
#if defined(__linux__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef FICLONE
#define FICLONE _IOW(0x94, 9, int)
#endif
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

static int copy_file(const char *in, const char *out)
{
#if defined(__linux__)
    {
        const int a = open(in, O_RDONLY);
        if (a >= 0) {
            const int b = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (b >= 0) {
                int ok = ioctl(b, FICLONE, a) == 0; /* reflink (btrfs, xfs) */
                if (!ok) {
                    struct stat stt;
                    if (!fstat(a, &stt)) {
                        off_t left = stt.st_size;
                        ok = 1;
                        while (left > 0) {
                            const ssize_t n = copy_file_range(a, NULL, b, NULL, (size_t)left, 0);
                            if (n <= 0) {
                                ok = left == stt.st_size ? -1 : 0; /* -1: not supported, fall back */
                                break;
                            }
                            left -= n;
                        }
                    }
                }
                close(a);
                close(b);
                if (ok == 1)
                    return 0;
                if (ok == 0)
                    return -1;
            } else {
                close(a);
                return -1;
            }
        }
    }
#endif
    FILE *a = fopen(in, "rb"), *b = fopen(out, "wb");
    if (!a || !b) {
        if (a)
            fclose(a);
        if (b)
            fclose(b);
        return -1;
    }
    const size_t bs = 1 << 22;
    char *buf = malloc(bs);
    int ret = buf ? 0 : -1;
    while (!ret) {
        const size_t n = fread(buf, 1, bs, a);
        if (n && fwrite(buf, 1, n, b) != n)
            ret = -1;
        if (n < bs) {
            if (ferror(a))
                ret = -1;
            break;
        }
    }
    free(buf);
    fclose(a);
    if (fclose(b))
        ret = -1;
    return ret;
}

static int cmp_loc(const void *a, const void *b)
{
    const Av1fgLoc *x = a, *y = b;
    return x->byte < y->byte ? -1 : x->byte > y->byte;
}

/* write the new seeds into f (opened r+b); returns the number of fields written */
static int64_t apply_patches(FILE *f, const Av1fgScan *s, int64_t **patched_bytes, size_t *n_patched)
{
    Av1fgLoc *l = malloc((s->n_locs + 1) * sizeof(*l));
    int64_t *pb = malloc((s->n_locs * 3 + 1) * sizeof(*pb));
    if (!l || !pb) {
        free(l);
        free(pb);
        return -1;
    }
    size_t n = 0, np = 0;
    for (size_t i = 0; i < s->n_locs; i++) {
        const Av1fgFrame *fr = &s->frames[s->locs[i].frame];
        if (fr->new_seed != fr->seed)
            l[n++] = s->locs[i];
    }
    qsort(l, n, sizeof(*l), cmp_loc);
    int64_t written = 0;
    for (size_t i = 0; i < n; i++) {
        const Av1fgFrame *fr = &s->frames[l[i].frame];
        uint8_t b[3];
        const size_t nb = l[i].bit ? 3 : 2;
        if (av1fg_read_at(f, l[i].byte, b, nb) != nb)
            goto fail;
        if (av1_get_bits16(b, l[i].bit) != fr->seed) {
            fprintf(stderr, "error: seed at %lld is %u, expected %u\n", (long long)l[i].byte,
                    av1_get_bits16(b, l[i].bit), fr->seed);
            goto fail;
        }
        av1_put_bits16(b, l[i].bit, fr->new_seed);
        if (fseek64(f, l[i].byte, SEEK_SET) || fwrite(b, 1, nb, f) != nb)
            goto fail;
        for (size_t k = 0; k < nb; k++)
            pb[np++] = l[i].byte + (int64_t)k;
        written++;
    }
    if (fflush(f))
        goto fail;
    free(l);
    *patched_bytes = pb;
    *n_patched = np;
    return written;
fail:
    free(l);
    free(pb);
    return -1;
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
    const char *in = NULL, *out = NULL, *set_seeds = NULL, *rank_all = NULL;
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
    if (so.candidates < 1 || so.candidates > 256 || so.bad_pct <= 0 || so.bad_pct >= 100 || so.threads < 0) {
        fprintf(stderr, "--candidates / --bad / --threads out of range\n");
        return 2;
    }
    const int write = out || in_place;
    if (!write && !dump && !dump_display)
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
    if (analyze || write) {
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
        if (!quiet || analyze)
            av1fg_print_report(analyze ? stdout : stderr, &s, &so, &rep, write);
        msg("seed scoring: %.2f s\n", now_s() - t1);
        av1fg_report_free(&rep);
    }
    if (write) {
        const char *target = in_place ? in : out;
        if (!in_place) {
            const double t2 = now_s();
            if (copy_file(in, out) < 0) {
                fprintf(stderr, "error: cannot copy %s to %s: %s\n", in, out, strerror(errno));
                av1fg_scan_free(&s);
                return 1;
            }
            msg("copied in %.2f s\n", now_s() - t2);
        }
        FILE *w = fopen(target, "r+b");
        if (!w) {
            fprintf(stderr, "error: %s: %s\n", target, strerror(errno));
            av1fg_scan_free(&s);
            return 1;
        }
        int64_t *pb = NULL;
        size_t npb = 0;
        const int64_t nw = apply_patches(w, &s, &pb, &npb);
        int ncrc = 0;
        if (nw >= 0)
            ncrc = av1fg_fix_crcs(w, &s.cont, pb, npb);
        free(pb);
        if (fclose(w) || nw < 0 || ncrc < 0) {
            fprintf(stderr, "error: writing %s failed%s\n", target, in_place ? " (the input may be partly patched)" : "");
            av1fg_scan_free(&s);
            return 1;
        }
        msg("%s: wrote %lld seed fields%s%s\n", target, (long long)nw, ncrc ? ", updated CRC-32 elements: " : "",
            ncrc ? "yes" : "");
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
