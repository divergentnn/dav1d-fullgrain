/*
 * av1grain: split an AV1 stream into a grain-free stream and the film grain
 * parameters it carried, for players that decode the grain-free stream with
 * any (hardware) decoder and draw the grain themselves (av1-fullgrain's
 * WebGPU player). Also decodes, with or without grain, for the tests.
 *
 *   av1grain strip  IN.ivf OUT.ivf PARAMS.json [--config SEQHDR.obu] [--threads N]
 *   av1grain decode IN.ivf OUT.yuv|- [--grain off|standard|full]
 *                   [--params PARAMS.json] [--config SEQHDR.obu] [--threads N]
 *                   [--frames N]
 *
 * strip: every frame header's film_grain_params() becomes apply_grain = 0
 *   (one bit; the rest of the grain syntax is dropped, the header is
 *   byte-aligned again and the OBU size rewritten). Tile data, sequence
 *   headers and everything else are copied unchanged, so the stream decodes
 *   to exactly the pictures the original gives with grain off.
 *   Redundant frame headers are dropped (they are optional).
 *   PARAMS.json lists the grain parameters dav1d applied to every OUTPUT
 *   picture (after load_grain_params / update_grain = 0 / show_existing_frame
 *   are resolved), in output order: "sets" holds the distinct parameter sets
 *   without the seed, "grain" one [set, random_seed] per picture ([-1, 0]: no
 *   grain on that picture).
 *   Positions come from dav1d's own frame header parser (the fg_hook in
 *   src/internal.h), so every header the decoder understands is handled.
 *
 * decode: writes the pictures as packed planes (Y, U, V; 16-bit little-endian
 *   samples above 8 bits) for hashing. --params applies the grain from a
 *   PARAMS.json to a stream that has none (a stripped one), with the dav1d
 *   synthesis selected by --grain (full = this fork's full-frame grain).
 *
 * IVF in and out (the MP4 wrapper is tools/av1grain_mp4.py). --config feeds
 * a sequence header (an MP4's av1C configOBUs) before the first frame; strip
 * does not copy it into OUT.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "config.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dav1d/dav1d.h"
#include "src/internal.h"

#define FAIL(...) do { fprintf(stderr, "av1grain: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

/* ------------------------------------------------------------------------ */
/* files                                                                      */

typedef struct TU {
    uint8_t *data;
    size_t sz;
    uint64_t pts;
} TU;

typedef struct Ivf {
    uint8_t hdr[32];
    TU *tu;
    int n;
} Ivf;

static uint8_t *read_file(const char *path, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) FAIL("cannot open %s: %s", path, strerror(errno));
    size_t cap = 1 << 20, n = 0;
    uint8_t *b = malloc(cap);
    for (;;) {
        if (n == cap) b = realloc(b, cap *= 2);
        if (!b) FAIL("out of memory");
        const size_t r = fread(b + n, 1, cap - n, f);
        if (!r) break;
        n += r;
    }
    fclose(f);
    *sz = n;
    return b;
}

static uint32_t rl32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t) p[3] << 24; }
static void wl32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }

static void read_ivf(const char *path, Ivf *ivf) {
    size_t sz;
    uint8_t *b = read_file(path, &sz);
    if (sz < 32 || memcmp(b, "DKIF", 4)) FAIL("%s: not an IVF file", path);
    memcpy(ivf->hdr, b, 32);
    const size_t hl = b[6] | b[7] << 8;
    size_t off = hl < 32 ? 32 : hl;
    int cap = 256;
    ivf->tu = malloc(cap * sizeof(TU));
    ivf->n = 0;
    while (off + 12 <= sz) {
        const size_t fs = rl32(b + off);
        uint64_t pts = rl32(b + off + 4) | (uint64_t) rl32(b + off + 8) << 32;
        off += 12;
        if (fs > sz - off) FAIL("%s: truncated frame %d", path, ivf->n);
        if (ivf->n == cap) ivf->tu = realloc(ivf->tu, (cap *= 2) * sizeof(TU));
        TU *t = &ivf->tu[ivf->n++];
        t->data = malloc(fs ? fs : 1);
        memcpy(t->data, b + off, fs);
        t->sz = fs;
        t->pts = pts;
        off += fs;
    }
    free(b);
}

static void write_ivf(const char *path, const Ivf *ivf) {
    FILE *f = fopen(path, "wb");
    if (!f) FAIL("cannot write %s: %s", path, strerror(errno));
    uint8_t h[32];
    memcpy(h, ivf->hdr, 32);
    wl32(h + 24, ivf->n);
    fwrite(h, 1, 32, f);
    for (int i = 0; i < ivf->n; i++) {
        uint8_t fh[12];
        wl32(fh, (uint32_t) ivf->tu[i].sz);
        wl32(fh + 4, (uint32_t) ivf->tu[i].pts);
        wl32(fh + 8, (uint32_t) (ivf->tu[i].pts >> 32));
        fwrite(fh, 1, 12, f);
        fwrite(ivf->tu[i].data, 1, ivf->tu[i].sz, f);
    }
    if (fclose(f)) FAIL("cannot write %s", path);
}

/* ------------------------------------------------------------------------ */
/* OBU syntax                                                                 */

static int leb128_read(const uint8_t *p, size_t avail, uint64_t *v) {
    *v = 0;
    for (int i = 0; i < 8; i++) {
        if ((size_t) i >= avail) return -1;
        *v |= (uint64_t) (p[i] & 0x7f) << (7 * i);
        if (!(p[i] & 0x80)) return i + 1;
    }
    return -1;
}

static int leb128_write(uint8_t *p, uint64_t v) {
    int n = 0;
    do {
        p[n] = v & 0x7f;
        v >>= 7;
        if (v) p[n] |= 0x80;
        n++;
    } while (v);
    return n;
}

typedef struct Obu {
    int type, hdr_len, has_size;  /* hdr_len: header byte(s), without the size field */
    size_t payload_off, payload_sz, total;  /* from the OBU's first byte */
} Obu;

static int obu_parse(const uint8_t *p, size_t avail, Obu *o) {
    if (avail < 1) return -1;
    o->type = (p[0] >> 3) & 15;
    const int ext = (p[0] >> 2) & 1;
    o->has_size = (p[0] >> 1) & 1;
    o->hdr_len = 1 + ext;
    if ((size_t) o->hdr_len > avail) return -1;
    if (o->has_size) {
        uint64_t len;
        const int n = leb128_read(p + o->hdr_len, avail - o->hdr_len, &len);
        if (n < 0 || len > avail - o->hdr_len - n) return -1;
        o->payload_off = o->hdr_len + n;
        o->payload_sz = (size_t) len;
    } else {
        o->payload_off = o->hdr_len;
        o->payload_sz = avail - o->hdr_len;
    }
    o->total = o->payload_off + o->payload_sz;
    return 0;
}

typedef struct BitW {
    uint8_t *buf;
    size_t bits;
} BitW;

static void put_bit(BitW *w, int b) {
    if (!(w->bits & 7)) w->buf[w->bits >> 3] = 0;
    if (b) w->buf[w->bits >> 3] |= 0x80 >> (w->bits & 7);
    w->bits++;
}

static int get_bit_at(const uint8_t *p, size_t pos) {
    return (p[pos >> 3] >> (7 - (pos & 7))) & 1;
}

/* ------------------------------------------------------------------------ */
/* the parser hook: where each frame header's grain syntax is                 */

typedef struct HdrPos {
    const uint8_t *obu;
    int type, fg_bitpos, end_bitpos;
} HdrPos;

typedef struct Hook {
    HdrPos *v;
    int n, cap;
} Hook;

static void hook_cb(void *cookie, const uint8_t *obu, int type, int fg_bitpos, int end_bitpos) {
    Hook *h = cookie;
    if (h->n == h->cap) h->v = realloc(h->v, (h->cap = h->cap ? 2 * h->cap : 256) * sizeof(HdrPos));
    if (!h->v) FAIL("out of memory");
    h->v[h->n++] = (HdrPos) { obu, type, fg_bitpos, end_bitpos };
}

/* the records come in stream order (dav1d parses OBUs in order); redundant
 * frame headers are usually skipped by the parser and have none */
static const HdrPos *hook_next(const Hook *h, int *cur, const uint8_t *obu) {
    if (*cur < h->n && h->v[*cur].obu == obu) return &h->v[(*cur)++];
    return NULL;
}

/* ------------------------------------------------------------------------ */
/* decoding                                                                   */

static void nofree(const uint8_t *buf, void *cookie) { (void) buf; (void) cookie; }

typedef void (*PicFn)(void *ctx, Dav1dContext *c, Dav1dPicture *p);

typedef struct Dec {
    Dav1dContext *c;
    PicFn fn;
    void *fnctx;
    int npics;
} Dec;

static void dec_open(Dec *d, int threads, int apply_grain, const char *mode, PicFn fn, void *fnctx) {
    /* read by dav1d_open (src/fg_variant.c) */
    setenv("DAV1D_GRAIN_FORCE", "0", 1);
    setenv("DAV1D_GRAIN_MODE", mode ? mode : "standard", 1);
    unsetenv("DAV1D_GRAIN_MODE_FILE");
    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = threads;
    s.max_frame_delay = 0;
    s.apply_grain = apply_grain;
    s.all_layers = 0;
    if (dav1d_open(&d->c, &s) < 0) FAIL("dav1d_open failed");
    d->fn = fn;
    d->fnctx = fnctx;
    d->npics = 0;
}

static int dec_pull(Dec *d) {
    for (;;) {
        Dav1dPicture p;
        memset(&p, 0, sizeof(p));
        const int r = dav1d_get_picture(d->c, &p);
        if (r == DAV1D_ERR(EAGAIN)) return 0;
        if (r < 0) return r;
        d->fn(d->fnctx, d->c, &p);
        dav1d_picture_unref(&p);
        d->npics++;
    }
}

/* data must stay valid until the decoder is closed (the hook keeps pointers) */
static void dec_send(Dec *d, const uint8_t *buf, size_t sz, uint64_t pts) {
    Dav1dData data;
    memset(&data, 0, sizeof(data));
    if (!sz) return;
    if (dav1d_data_wrap(&data, buf, sz, nofree, NULL) < 0) FAIL("dav1d_data_wrap failed");
    data.m.timestamp = (int64_t) pts;
    while (data.sz) {
        const int r = dav1d_send_data(d->c, &data);
        if (r == DAV1D_ERR(EAGAIN)) {
            if (dec_pull(d) < 0) FAIL("decode error");
            continue;
        }
        if (r < 0) FAIL("decode error %d", r);
    }
    if (dec_pull(d) < 0) FAIL("decode error");
}

static void dec_drain(Dec *d) {
    for (;;) {
        Dav1dPicture p;
        memset(&p, 0, sizeof(p));
        const int r = dav1d_get_picture(d->c, &p);
        if (r == DAV1D_ERR(EAGAIN)) break;
        if (r < 0) FAIL("decode error %d while draining", r);
        d->fn(d->fnctx, d->c, &p);
        dav1d_picture_unref(&p);
        d->npics++;
    }
}

/* ------------------------------------------------------------------------ */
/* grain parameter sets                                                       */

static int has_grain(const Dav1dFrameHeader *h) {
    const Dav1dFilmGrainData *g = &h->film_grain.data;
    return h->film_grain.present &&
           (g->num_y_points || g->num_uv_points[0] || g->num_uv_points[1] ||
            (g->clip_to_restricted_range && g->chroma_scaling_from_luma));
}

/* the parameter set without the seed, canonical (unused entries zero) */
static Dav1dFilmGrainData canon(const Dav1dFilmGrainData *g) {
    Dav1dFilmGrainData c;
    memset(&c, 0, sizeof(c));
    c.num_y_points = g->num_y_points;
    for (int i = 0; i < g->num_y_points; i++) memcpy(c.y_points[i], g->y_points[i], 2);
    c.chroma_scaling_from_luma = g->chroma_scaling_from_luma;
    for (int pl = 0; pl < 2; pl++) {
        c.num_uv_points[pl] = g->num_uv_points[pl];
        for (int i = 0; i < g->num_uv_points[pl]; i++) memcpy(c.uv_points[pl][i], g->uv_points[pl][i], 2);
        c.uv_mult[pl] = g->uv_mult[pl];
        c.uv_luma_mult[pl] = g->uv_luma_mult[pl];
        c.uv_offset[pl] = g->uv_offset[pl];
    }
    c.scaling_shift = g->scaling_shift;
    c.ar_coeff_lag = g->ar_coeff_lag;
    const int npos = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
    memcpy(c.ar_coeffs_y, g->ar_coeffs_y, npos);
    for (int pl = 0; pl < 2; pl++) memcpy(c.ar_coeffs_uv[pl], g->ar_coeffs_uv[pl], npos + 1);
    c.ar_coeff_shift = g->ar_coeff_shift;
    c.grain_scale_shift = g->grain_scale_shift;
    c.overlap_flag = g->overlap_flag;
    c.clip_to_restricted_range = g->clip_to_restricted_range;
    return c;
}

typedef struct Params {
    Dav1dFilmGrainData *sets;
    int nsets, capsets;
    int (*grain)[2];
    int64_t *pts;
    int n, cap;
    int w, h, bpc, layout, mtrx, trc, pri, range;
} Params;

static void params_add(Params *P, const Dav1dPicture *p) {
    if (P->n == P->cap) {
        P->cap = P->cap ? 2 * P->cap : 256;
        P->grain = realloc(P->grain, P->cap * sizeof(*P->grain));
        P->pts = realloc(P->pts, P->cap * sizeof(*P->pts));
        if (!P->grain || !P->pts) FAIL("out of memory");
    }
    if (!P->n) {
        P->w = p->p.w; P->h = p->p.h; P->bpc = p->p.bpc; P->layout = p->p.layout;
        P->mtrx = p->seq_hdr->mtrx; P->trc = p->seq_hdr->trc; P->pri = p->seq_hdr->pri;
        P->range = p->seq_hdr->color_range;
    } else if (p->p.w != P->w || p->p.h != P->h || p->p.bpc != (int) P->bpc || (int) p->p.layout != P->layout) {
        FAIL("picture %d: the format changes (%dx%d %d-bit layout %d); not supported",
             P->n, p->p.w, p->p.h, p->p.bpc, p->p.layout);
    }
    int set = -1, seed = 0;
    if (has_grain(p->frame_hdr)) {
        const Dav1dFilmGrainData c = canon(&p->frame_hdr->film_grain.data);
        for (int i = 0; i < P->nsets; i++)
            if (!memcmp(&P->sets[i], &c, sizeof(c))) { set = i; break; }
        if (set < 0) {
            if (P->nsets == P->capsets) {
                P->capsets = P->capsets ? 2 * P->capsets : 16;
                P->sets = realloc(P->sets, P->capsets * sizeof(*P->sets));
                if (!P->sets) FAIL("out of memory");
            }
            P->sets[set = P->nsets++] = c;
        }
        seed = p->frame_hdr->film_grain.data.seed;
    }
    P->grain[P->n][0] = set;
    P->grain[P->n][1] = seed;
    P->pts[P->n] = p->m.timestamp;
    P->n++;
}

static void json_ints(FILE *f, const int *v, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) fprintf(f, "%s%d", i ? "," : "", v[i]);
    fputc(']', f);
}

static void json_points(FILE *f, const uint8_t (*pt)[2], int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) fprintf(f, "%s[%d,%d]", i ? "," : "", pt[i][0], pt[i][1]);
    fputc(']', f);
}

static void write_params(const char *path, const Params *P) {
    FILE *f = fopen(path, "wb");
    if (!f) FAIL("cannot write %s: %s", path, strerror(errno));
    fprintf(f, "{\"format\":\"av1-fullgrain-params\",\"version\":1,\n"
               "\"width\":%d,\"height\":%d,\"bitdepth\":%d,\"layout\":%d,"
               "\"mtrx\":%d,\"trc\":%d,\"pri\":%d,\"color_range\":%d,\"frames\":%d,\n\"sets\":[",
            P->w, P->h, P->bpc, P->layout, P->mtrx, P->trc, P->pri, P->range, P->n);
    for (int s = 0; s < P->nsets; s++) {
        const Dav1dFilmGrainData *g = &P->sets[s];
        const int npos = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
        int v[28];
        fprintf(f, "%s\n{\"num_y_points\":%d,\"y_points\":", s ? "," : "", g->num_y_points);
        json_points(f, (const uint8_t (*)[2]) g->y_points, g->num_y_points);
        fprintf(f, ",\"chroma_scaling_from_luma\":%d,\"num_uv_points\":[%d,%d],\"uv_points\":[",
                g->chroma_scaling_from_luma, g->num_uv_points[0], g->num_uv_points[1]);
        json_points(f, (const uint8_t (*)[2]) g->uv_points[0], g->num_uv_points[0]);
        fputc(',', f);
        json_points(f, (const uint8_t (*)[2]) g->uv_points[1], g->num_uv_points[1]);
        fprintf(f, "],\"scaling_shift\":%d,\"ar_coeff_lag\":%d,\"ar_coeffs_y\":",
                (int) g->scaling_shift, g->ar_coeff_lag);
        for (int i = 0; i < npos; i++) v[i] = g->ar_coeffs_y[i];
        json_ints(f, v, npos);
        fputs(",\"ar_coeffs_uv\":[", f);
        for (int pl = 0; pl < 2; pl++) {
            for (int i = 0; i <= npos; i++) v[i] = g->ar_coeffs_uv[pl][i];
            if (pl) fputc(',', f);
            json_ints(f, v, npos + 1);
        }
        fprintf(f, "],\"ar_coeff_shift\":%d,\"grain_scale_shift\":%d,"
                   "\"uv_mult\":[%d,%d],\"uv_luma_mult\":[%d,%d],\"uv_offset\":[%d,%d],"
                   "\"overlap_flag\":%d,\"clip_to_restricted_range\":%d}",
                (int) g->ar_coeff_shift, g->grain_scale_shift, g->uv_mult[0], g->uv_mult[1],
                g->uv_luma_mult[0], g->uv_luma_mult[1], g->uv_offset[0], g->uv_offset[1],
                g->overlap_flag, g->clip_to_restricted_range);
    }
    fputs("],\n\"grain\":[", f);
    for (int i = 0; i < P->n; i++)
        fprintf(f, "%s[%d,%d]", i ? (i % 16 ? "," : ",\n") : "", P->grain[i][0], P->grain[i][1]);
    fputs("],\n\"pts\":[", f);
    for (int i = 0; i < P->n; i++) fprintf(f, "%s%" PRId64, i ? "," : "", P->pts[i]);
    fputs("]}\n", f);
    if (fclose(f)) FAIL("cannot write %s", path);
}

/* ------------------------------------------------------------------------ */
/* a small JSON reader (enough for PARAMS.json)                               */

typedef struct J {
    int type;          /* 'o' object, 'a' array, 'n' number, 's' string, 'l' literal */
    double num;
    char *str;         /* string value, or key when inside an object */
    struct J *kids;
    int nkids;
    char *key;
} J;

static const char *jp_ws(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++; return p; }

static const char *jp_val(const char *p, J *v);

static const char *jp_str(const char *p, char **out) {
    if (*p != '"') FAIL("params: string expected");
    const char *e = ++p;
    while (*e && *e != '"') { if (*e == '\\' && e[1]) e++; e++; }
    if (!*e) FAIL("params: unterminated string");
    *out = strndup(p, e - p);
    return e + 1;
}

static const char *jp_val(const char *p, J *v) {
    memset(v, 0, sizeof(*v));
    p = jp_ws(p);
    if (*p == '{' || *p == '[') {
        const int obj = *p == '{';
        v->type = obj ? 'o' : 'a';
        p = jp_ws(p + 1);
        int cap = 0;
        if (*p == (obj ? '}' : ']')) return p + 1;
        for (;;) {
            if (v->nkids == cap) v->kids = realloc(v->kids, (cap = cap ? 2 * cap : 8) * sizeof(J));
            char *key = NULL;
            if (obj) {
                p = jp_str(jp_ws(p), &key);
                p = jp_ws(p);
                if (*p++ != ':') FAIL("params: ':' expected");
            }
            p = jp_val(p, &v->kids[v->nkids]);
            v->kids[v->nkids++].key = key;
            p = jp_ws(p);
            if (*p == ',') { p++; continue; }
            if (*p == (obj ? '}' : ']')) return p + 1;
            FAIL("params: ',' expected");
        }
    }
    if (*p == '"') { v->type = 's'; return jp_str(p, &v->str); }
    if (*p == '-' || (*p >= '0' && *p <= '9')) {
        char *e;
        v->type = 'n';
        v->num = strtod(p, &e);
        return e;
    }
    if (!strncmp(p, "true", 4)) { v->type = 'l'; v->num = 1; return p + 4; }
    if (!strncmp(p, "false", 5)) { v->type = 'l'; return p + 5; }
    if (!strncmp(p, "null", 4)) { v->type = 'l'; return p + 4; }
    FAIL("params: unexpected '%.10s'", p);
}

static const J *jget(const J *o, const char *key) {
    for (int i = 0; i < o->nkids; i++)
        if (o->kids[i].key && !strcmp(o->kids[i].key, key)) return &o->kids[i];
    FAIL("params: no \"%s\"", key);
}

static int jint(const J *o, const char *key) {
    const J *v = jget(o, key);
    if (v->type != 'n' && v->type != 'l') FAIL("params: \"%s\" is not a number", key);
    return (int) v->num;
}

static const J *jarr(const J *o, const char *key, int n) {
    const J *v = jget(o, key);
    if (v->type != 'a' || (n >= 0 && v->nkids != n)) FAIL("params: \"%s\" is not an array of %d", key, n);
    return v;
}

static void read_params(const char *path, Params *P) {
    size_t sz;
    char *txt = (char *) read_file(path, &sz);
    txt = realloc(txt, sz + 1);
    txt[sz] = 0;
    J root;
    jp_val(txt, &root);
    if (root.type != 'o') FAIL("params: not an object");
    if (strcmp(jget(&root, "format")->str ? jget(&root, "format")->str : "", "av1-fullgrain-params") ||
        jint(&root, "version") != 1)
        FAIL("%s: not av1-fullgrain-params version 1", path);
    memset(P, 0, sizeof(*P));
    P->w = jint(&root, "width"); P->h = jint(&root, "height"); P->bpc = jint(&root, "bitdepth");
    P->layout = jint(&root, "layout"); P->mtrx = jint(&root, "mtrx");
    const J *sets = jarr(&root, "sets", -1);
    P->nsets = sets->nkids;
    P->sets = calloc(P->nsets ? P->nsets : 1, sizeof(*P->sets));
    for (int s = 0; s < P->nsets; s++) {
        const J *o = &sets->kids[s];
        Dav1dFilmGrainData *g = &P->sets[s];
        g->num_y_points = jint(o, "num_y_points");
        if (g->num_y_points > 14) FAIL("params: num_y_points");
        const J *yp = jarr(o, "y_points", g->num_y_points);
        for (int i = 0; i < g->num_y_points; i++)
            for (int k = 0; k < 2; k++) {
                if (yp->kids[i].type != 'a' || yp->kids[i].nkids != 2) FAIL("params: y_points");
                g->y_points[i][k] = (uint8_t) yp->kids[i].kids[k].num;
            }
        g->chroma_scaling_from_luma = jint(o, "chroma_scaling_from_luma");
        const J *nuv = jarr(o, "num_uv_points", 2), *uvp = jarr(o, "uv_points", 2);
        for (int pl = 0; pl < 2; pl++) {
            g->num_uv_points[pl] = (int) nuv->kids[pl].num;
            if (g->num_uv_points[pl] > 10 || uvp->kids[pl].nkids != g->num_uv_points[pl]) FAIL("params: uv_points");
            for (int i = 0; i < g->num_uv_points[pl]; i++)
                for (int k = 0; k < 2; k++) {
                    if (uvp->kids[pl].kids[i].nkids != 2) FAIL("params: uv_points");
                    g->uv_points[pl][i][k] = (uint8_t) uvp->kids[pl].kids[i].kids[k].num;
                }
        }
        g->scaling_shift = jint(o, "scaling_shift");
        g->ar_coeff_lag = jint(o, "ar_coeff_lag");
        if (g->ar_coeff_lag < 0 || g->ar_coeff_lag > 3) FAIL("params: ar_coeff_lag");
        const int npos = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
        const J *ay = jarr(o, "ar_coeffs_y", npos), *auv = jarr(o, "ar_coeffs_uv", 2);
        for (int i = 0; i < npos; i++) g->ar_coeffs_y[i] = (int8_t) ay->kids[i].num;
        for (int pl = 0; pl < 2; pl++) {
            if (auv->kids[pl].nkids != npos + 1) FAIL("params: ar_coeffs_uv");
            for (int i = 0; i <= npos; i++) g->ar_coeffs_uv[pl][i] = (int8_t) auv->kids[pl].kids[i].num;
        }
        g->ar_coeff_shift = jint(o, "ar_coeff_shift");
        g->grain_scale_shift = jint(o, "grain_scale_shift");
        const J *um = jarr(o, "uv_mult", 2), *ulm = jarr(o, "uv_luma_mult", 2), *uo = jarr(o, "uv_offset", 2);
        for (int pl = 0; pl < 2; pl++) {
            g->uv_mult[pl] = (int) um->kids[pl].num;
            g->uv_luma_mult[pl] = (int) ulm->kids[pl].num;
            g->uv_offset[pl] = (int) uo->kids[pl].num;
        }
        g->overlap_flag = jint(o, "overlap_flag");
        g->clip_to_restricted_range = jint(o, "clip_to_restricted_range");
    }
    const J *gr = jarr(&root, "grain", -1);
    P->n = gr->nkids;
    P->grain = calloc(P->n ? P->n : 1, sizeof(*P->grain));
    for (int i = 0; i < P->n; i++) {
        if (gr->kids[i].type != 'a' || gr->kids[i].nkids != 2) FAIL("params: grain[%d]", i);
        P->grain[i][0] = (int) gr->kids[i].kids[0].num;
        P->grain[i][1] = (int) gr->kids[i].kids[1].num;
        if (P->grain[i][0] >= P->nsets) FAIL("params: grain[%d] names set %d", i, P->grain[i][0]);
    }
    free(txt);
}

/* ------------------------------------------------------------------------ */
/* strip                                                                      */

static void strip_pic(void *ctx, Dav1dContext *c, Dav1dPicture *p) {
    (void) c;
    params_add(ctx, p);
}

/* rewrite one frame header OBU: apply_grain = 0 at fg_bitpos, then trailing
 * bits (OBU_FRAME_HDR / OBU_REDUNDANT_FRAME_HDR) or byte alignment + the
 * original tile group (OBU_FRAME) -> bytes written to out */
static size_t rewrite_hdr(uint8_t *out, const uint8_t *obu, const Obu *o, const HdrPos *hp) {
    uint8_t *pay = malloc(o->payload_sz + 16);
    BitW w = { pay, 0 };
    const size_t start = o->payload_off * 8;
    if ((size_t) hp->fg_bitpos < start || (size_t) hp->end_bitpos > o->total * 8 || hp->fg_bitpos > hp->end_bitpos)
        FAIL("frame header bit positions out of range");
    for (size_t b = start; b < (size_t) hp->fg_bitpos; b++) put_bit(&w, get_bit_at(obu, b));
    put_bit(&w, 0);                                   /* apply_grain = 0 */
    if (o->type == DAV1D_OBU_FRAME) {
        while (w.bits & 7) put_bit(&w, 0);            /* byte_alignment() */
        const size_t tiles = ((size_t) hp->end_bitpos + 7) >> 3;
        memcpy(pay + (w.bits >> 3), obu + tiles, o->total - tiles);
        w.bits += (o->total - tiles) * 8;
    } else {
        put_bit(&w, 1);                               /* trailing_bits() */
        while (w.bits & 7) put_bit(&w, 0);
    }
    const size_t psz = w.bits >> 3;
    size_t n = 0;
    out[n++] = obu[0] | 2;                            /* obu_has_size_field = 1 */
    if (o->hdr_len == 2) out[n++] = obu[1];
    n += leb128_write(out + n, psz);
    memcpy(out + n, pay, psz);
    free(pay);
    return n + psz;
}

static int cmd_strip(int argc, char **argv) {
    const char *in = NULL, *out = NULL, *pj = NULL, *cfg = NULL;
    int threads = 4;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!in) in = argv[i];
        else if (!out) out = argv[i];
        else if (!pj) pj = argv[i];
        else FAIL("strip: unexpected %s", argv[i]);
    }
    if (!pj) FAIL("usage: av1grain strip IN.ivf OUT.ivf PARAMS.json [--config SEQHDR.obu] [--threads N]");
    Ivf ivf;
    read_ivf(in, &ivf);
    Params P;
    memset(&P, 0, sizeof(P));
    Hook hook = { 0 };
    Dec d;
    dec_open(&d, threads, 0, "standard", strip_pic, &P);
    d.c->fg_hook.cb = hook_cb;
    d.c->fg_hook.cookie = &hook;
    size_t cfgsz = 0;
    uint8_t *cfgbuf = cfg ? read_file(cfg, &cfgsz) : NULL;
    for (int i = 0; i < ivf.n; i++) {
        if (!i && cfgsz) dec_send(&d, cfgbuf, cfgsz, ivf.tu[0].pts);
        dec_send(&d, ivf.tu[i].data, ivf.tu[i].sz, ivf.tu[i].pts);
    }
    dec_drain(&d);

    /* rewrite */
    Ivf o = ivf;
    o.tu = malloc(ivf.n * sizeof(TU));
    int nhdr = 0, ndrop = 0, nfg = 0, cur = 0;
    size_t in_bytes = 0, out_bytes = 0;
    for (int i = 0; i < ivf.n; i++) {
        const TU *t = &ivf.tu[i];
        uint8_t *nb = malloc(2 * t->sz + 64);
        size_t n = 0, off = 0;
        int k = 0;
        while (off < t->sz) {
            Obu ob;
            if (obu_parse(t->data + off, t->sz - off, &ob) < 0) FAIL("frame %d: bad OBU at byte %zu", i, off);
            const uint8_t *obu = t->data + off;
            const int is_hdr = ob.type == DAV1D_OBU_FRAME || ob.type == DAV1D_OBU_FRAME_HDR ||
                               ob.type == DAV1D_OBU_REDUNDANT_FRAME_HDR;
            const HdrPos *hp = is_hdr ? hook_next(&hook, &cur, obu) : NULL;
            if (is_hdr && !hp) {
                if (ob.type != DAV1D_OBU_REDUNDANT_FRAME_HDR)
                    FAIL("frame %d: the decoder did not parse frame header OBU %d (type %d)", i, k, ob.type);
                ndrop++;                               /* optional copy of a header */
            } else if (hp && hp->fg_bitpos >= 0) {
                n += rewrite_hdr(nb + n, obu, &ob, hp);
                nhdr++;
                nfg++;
            } else {
                if (hp) nhdr++;
                memcpy(nb + n, obu, ob.total);
                n += ob.total;
            }
            off += ob.total;
            k++;
        }
        o.tu[i] = (TU) { nb, n, t->pts };
        in_bytes += t->sz;
        out_bytes += n;
    }
    if (cur != hook.n) FAIL("%d of %d parsed frame headers were not found in the stream", hook.n - cur, hook.n);
    if (P.n != ivf.n)
        fprintf(stderr, "av1grain: note: %d frames in, %d pictures out\n", ivf.n, P.n);
    write_ivf(out, &o);
    write_params(pj, &P);
    int with = 0;
    for (int i = 0; i < P.n; i++) with += P.grain[i][0] >= 0;
    fprintf(stderr, "av1grain: %d frames, %d pictures (%d with grain, %d parameter set%s), "
                    "%d frame headers (%d grain syntax removed, %d redundant dropped), %zu -> %zu bytes\n",
            ivf.n, P.n, with, P.nsets, P.nsets == 1 ? "" : "s", nhdr, nfg, ndrop, in_bytes, out_bytes);
    dav1d_close(&d.c);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* decode                                                                     */

typedef struct Out {
    FILE *f;
    const Params *P;   /* apply these instead of the stream's grain */
    int n, limit;
} Out;

static void write_pic(FILE *f, const Dav1dPicture *p) {
    const int hbd = p->p.bpc > 8, bs = hbd ? 2 : 1;
    const int ssx = p->p.layout != DAV1D_PIXEL_LAYOUT_I444, ssy = p->p.layout == DAV1D_PIXEL_LAYOUT_I420;
    for (int pl = 0; pl < (p->p.layout == DAV1D_PIXEL_LAYOUT_I400 ? 1 : 3); pl++) {
        const int w = pl ? (p->p.w + ssx) >> ssx : p->p.w, h = pl ? (p->p.h + ssy) >> ssy : p->p.h;
        const uint8_t *row = p->data[pl];
        const ptrdiff_t stride = p->stride[pl ? 1 : 0];
        for (int y = 0; y < h; y++, row += stride)
            if (fwrite(row, bs, w, f) != (size_t) w) FAIL("write failed");
    }
}

static void decode_pic(void *ctx, Dav1dContext *c, Dav1dPicture *p) {
    Out *o = ctx;
    if (o->limit && o->n >= o->limit) return;
    const int idx = o->n++;
    if (!o->P) { write_pic(o->f, p); return; }
    if (idx >= o->P->n) FAIL("more pictures than the params list (%d)", o->P->n);
    if (has_grain(p->frame_hdr)) FAIL("picture %d already has grain; --params wants a stripped stream", idx);
    const int set = o->P->grain[idx][0];
    if (set < 0) { write_pic(o->f, p); return; }
    Dav1dFrameHeader fh = *p->frame_hdr;
    fh.film_grain.present = 1;
    fh.film_grain.update = 1;
    fh.film_grain.data = o->P->sets[set];
    fh.film_grain.data.seed = o->P->grain[idx][1];
    Dav1dPicture in = *p, out;
    in.frame_hdr = &fh;
    memset(&out, 0, sizeof(out));
    const int r = dav1d_apply_grain(c, &out, &in);
    if (r < 0) FAIL("dav1d_apply_grain failed: %d", r);
    write_pic(o->f, &out);
    dav1d_picture_unref(&out);
}

static int cmd_decode(int argc, char **argv) {
    const char *in = NULL, *out = NULL, *pj = NULL, *cfg = NULL, *grain = "full";
    int threads = 4, limit = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grain") && i + 1 < argc) grain = argv[++i];
        else if (!strcmp(argv[i], "--params") && i + 1 < argc) pj = argv[++i];
        else if (!in) in = argv[i];
        else if (!out) out = argv[i];
        else FAIL("decode: unexpected %s", argv[i]);
    }
    if (!out) FAIL("usage: av1grain decode IN.ivf OUT.yuv|- [--grain off|standard|full] [--params P.json] [--config F] [--threads N]");
    const int off = !strcmp(grain, "off");
    if (!off && strcmp(grain, "standard") && strcmp(grain, "full") && strcmp(grain, "dual") && strcmp(grain, "multi"))
        FAIL("--grain off|standard|full");
    if (off && pj) FAIL("--params goes with --grain standard|full");
    Params P;
    if (pj) read_params(pj, &P);
    Ivf ivf;
    read_ivf(in, &ivf);
    Out o = { strcmp(out, "-") ? fopen(out, "wb") : stdout, pj ? &P : NULL, 0, limit };
    if (!o.f) FAIL("cannot write %s", out);
    Dec d;
    dec_open(&d, threads, !off && !pj, off ? "standard" : grain, decode_pic, &o);
    size_t cfgsz = 0;
    uint8_t *cfgbuf = cfg ? read_file(cfg, &cfgsz) : NULL;
    for (int i = 0; i < ivf.n && !(limit && o.n >= limit); i++) {
        if (!i && cfgsz) dec_send(&d, cfgbuf, cfgsz, ivf.tu[0].pts);
        dec_send(&d, ivf.tu[i].data, ivf.tu[i].sz, ivf.tu[i].pts);
    }
    dec_drain(&d);
    if (pj && !limit && o.n != P.n) FAIL("%d pictures, but the params list %d", o.n, P.n);
    if (o.f != stdout && fclose(o.f)) FAIL("cannot write %s", out);
    if (o.f == stdout) fflush(stdout);
    fprintf(stderr, "av1grain: %d pictures\n", o.n);
    dav1d_close(&d.c);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "strip")) return cmd_strip(argc - 2, argv + 2);
    if (argc >= 2 && !strcmp(argv[1], "decode")) return cmd_decode(argc - 2, argv + 2);
    fprintf(stderr,
            "av1grain strip  IN.ivf OUT.ivf PARAMS.json [--config SEQHDR.obu] [--threads N]\n"
            "av1grain decode IN.ivf OUT.yuv|- [--grain off|standard|full] [--params PARAMS.json]\n"
            "                [--config SEQHDR.obu] [--threads N] [--frames N]\n");
    return 2;
}
