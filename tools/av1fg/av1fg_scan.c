/*
 * av1fg_scan: parse every AV1 frame header of a file and collect the grain
 * seeds, their file positions, the grain parameter sets and display order.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#if !defined(_WIN32) && !defined(_GNU_SOURCE) && !defined(__APPLE__)
#define _GNU_SOURCE /* posix_fadvise */
#endif
#include "av1fg_scan.h"

#if defined(__linux__)
#include <fcntl.h>
#endif
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define GROW(arr, n, cap, init)                                                   \
    do {                                                                          \
        if ((n) == (cap)) {                                                       \
            const size_t nc_ = (cap) ? (cap) * 2 : (init);                        \
            void *p_ = realloc((arr), nc_ * sizeof(*(arr)));                      \
            if (!p_)                                                              \
                return -1;                                                        \
            (arr) = p_;                                                           \
            (cap) = nc_;                                                          \
        }                                                                         \
    } while (0)

static void seterr(Av1fgScan *s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s->err, sizeof(s->err), fmt, ap);
    va_end(ap);
}

void av1fg_scan_free(Av1fgScan *s)
{
    av1fg_container_free(&s->cont);
    free(s->frames);
    free(s->locs);
    free(s->display);
    free(s->params);
    memset(s, 0, sizeof(*s));
}

static uint32_t hash_bytes(const void *p, size_t n)
{
    const uint8_t *b = p;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++)
        h = (h ^ b[i]) * 16777619u;
    return h;
}

static int param_id(Av1fgScan *s, const Av1FilmGrain *g, int w, int h)
{
    Av1fgParamSet key;
    memset(&key, 0, sizeof(key));
    key.fg = *g;
    key.fg.grain_seed = 0;
    key.fg.apply_grain = 0;
    key.fg.update_grain = 0;
    key.fg.film_grain_params_ref_idx = 0;
    key.bit_depth = s->seq.bit_depth;
    key.subsampling_x = s->seq.subsampling_x;
    key.subsampling_y = s->seq.subsampling_y;
    key.mono_chrome = s->seq.mono_chrome;
    const uint32_t hh = hash_bytes(&key.fg, sizeof(key.fg)) ^ (uint32_t)(key.bit_depth * 131 + key.subsampling_x * 7 +
                                                                        key.subsampling_y * 3 + key.mono_chrome);
    /* most streams reuse the last few sets: search backwards */
    for (size_t i = s->n_params; i-- > 0;) {
        const Av1fgParamSet *p = &s->params[i];
        if (p->hash == hh && !memcmp(&p->fg, &key.fg, sizeof(key.fg)) && p->bit_depth == key.bit_depth &&
            p->subsampling_x == key.subsampling_x && p->subsampling_y == key.subsampling_y &&
            p->mono_chrome == key.mono_chrome)
            return (int)i;
    }
    GROW(s->params, s->n_params, s->cap_params, 64);
    key.hash = hh;
    key.width = w;
    key.height = h;
    s->params[s->n_params] = key;
    return (int)s->n_params++;
}

static int add_display(Av1fgScan *s, int64_t fi)
{
    GROW(s->display, s->n_display, s->cap_display, 4096);
    Av1fgFrame *fr = &s->frames[fi];
    if (fr->display_idx < 0)
        fr->display_idx = (int64_t)s->n_display;
    fr->shown++;
    s->display[s->n_display++] = fi;
    if (fr->param >= 0)
        s->params[fr->param].frames++;
    return 0;
}

void av1fg_format_grain(char *buf, size_t len, const Av1FilmGrain *g, int with_seed)
{
    size_t o = 0;
#define P(...)                                                                    \
    do {                                                                          \
        if (o < len)                                                              \
            o += (size_t)snprintf(buf + o, len - o, __VA_ARGS__);                \
    } while (0)
    if (!g->apply_grain) {
        P("grain 0");
        return;
    }
    if (with_seed)
        P("seed %u ", g->grain_seed);
    P("y[");
    for (int i = 0; i < g->num_y_points; i++)
        P("%s%d:%d", i ? " " : "", g->point_y_value[i], g->point_y_scaling[i]);
    P("] csfl %d cb[", g->chroma_scaling_from_luma);
    for (int i = 0; i < g->num_cb_points; i++)
        P("%s%d:%d", i ? " " : "", g->point_cb_value[i], g->point_cb_scaling[i]);
    P("] cr[");
    for (int i = 0; i < g->num_cr_points; i++)
        P("%s%d:%d", i ? " " : "", g->point_cr_value[i], g->point_cr_scaling[i]);
    P("] scaling_shift %d lag %d", g->grain_scaling_minus_8 + 8, g->ar_coeff_lag);
    const int npl = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
    const int npc = npl + (g->num_y_points ? 1 : 0);
    P(" ary[");
    if (g->num_y_points)
        for (int i = 0; i < npl; i++)
            P("%s%d", i ? " " : "", g->ar_coeffs_y_plus_128[i] - 128);
    P("] arcb[");
    if (g->num_cb_points || g->chroma_scaling_from_luma)
        for (int i = 0; i < npc; i++)
            P("%s%d", i ? " " : "", g->ar_coeffs_cb_plus_128[i] - 128);
    P("] arcr[");
    if (g->num_cr_points || g->chroma_scaling_from_luma)
        for (int i = 0; i < npc; i++)
            P("%s%d", i ? " " : "", g->ar_coeffs_cr_plus_128[i] - 128);
    P("] ar_shift %d gss %d", g->ar_coeff_shift_minus_6 + 6, g->grain_scale_shift);
    if (g->num_cb_points)
        P(" cb %d %d %d", g->cb_mult - 128, g->cb_luma_mult - 128, g->cb_offset - 256);
    if (g->num_cr_points)
        P(" cr %d %d %d", g->cr_mult - 128, g->cr_luma_mult - 128, g->cr_offset - 256);
    P(" overlap %d clip %d", g->overlap_flag, g->clip_to_restricted_range);
#undef P
}

typedef struct ScanState {
    int64_t slot[8];    /* frame index held by each reference slot */
    int64_t cur;        /* frame being decoded, -1 if none */
} ScanState;

static int handle_obu(Av1fgScan *s, ScanState *st, const Av1Obu *obu, int64_t off, const Av1Parser *p)
{
    s->n_obus++;
    if (obu->dropped) {
        s->n_dropped++;
        return 0;
    }
    if (obu->type == AV1_OBU_SEQUENCE_HEADER) {
        s->seq = p->seq;
        s->have_seq = 1;
    }
    if (obu->type == AV1_OBU_REDUNDANT_FRAME_HEADER)
        s->n_redundant++;
    if (obu->has_frame_hdr) {
        const Av1FrameHdr *h = &obu->frame_hdr;
        s->n_headers++;
        int64_t fi;
        if (h->is_copy) {
            s->n_copies++;
            fi = st->cur;
            if (fi < 0)
                return 0;
        } else if (h->show_existing_frame) {
            s->n_show_existing++;
            const int64_t shown = st->slot[h->frame_to_show_map_idx];
            if (shown < 0) {
                seterr(s, "show_existing_frame of slot %d, which holds no frame", h->frame_to_show_map_idx);
                return -1;
            }
            if (add_display(s, shown) < 0)
                return -1;
            if (h->frame_type == AV1_KEY_FRAME)
                for (int i = 0; i < 8; i++)
                    st->slot[i] = shown;
            return 0;
        } else {
            GROW(s->frames, s->n_frames, s->cap_frames, 4096);
            fi = (int64_t)s->n_frames++;
            Av1fgFrame *fr = &s->frames[fi];
            memset(fr, 0, sizeof(*fr));
            fr->decode_idx = fi;
            fr->display_idx = -1;
            fr->frame_type = h->frame_type;
            fr->show_frame = h->show_frame;
            fr->showable_frame = h->showable_frame;
            fr->width = h->frame_width;
            fr->height = h->frame_height;
            fr->upscaled_width = h->upscaled_width;
            fr->apply_grain = h->fg.apply_grain;
            fr->update_grain_coded = h->update_grain_coded;
            fr->grain_ref_idx = h->update_grain_coded ? -1 : h->fg.film_grain_params_ref_idx;
            fr->seed = fr->new_seed = h->fg.grain_seed;
            fr->param = h->fg.apply_grain ? param_id(s, &h->fg, h->upscaled_width, h->frame_height) : -1;
            if (fr->param < 0 && h->fg.apply_grain)
                return -1;
            fr->loc = -1;
            fr->temporal_id = h->temporal_id;
            fr->spatial_id = h->spatial_id;
            st->cur = fi;
        }
        if (h->seed_bit_offset >= 0) {
            GROW(s->locs, s->n_locs, s->cap_locs, 4096);
            const int64_t bit = (int64_t)obu->header_len * 8 + h->seed_bit_offset;
            Av1fgLoc *l = &s->locs[s->n_locs];
            l->byte = off + (bit >> 3);
            l->bit = (uint8_t)(bit & 7);
            l->is_copy = (uint8_t)h->is_copy;
            l->frame = fi;
            Av1fgFrame *fr = &s->frames[fi];
            if (fr->loc < 0)
                fr->loc = (int64_t)s->n_locs;
            fr->n_loc++;
            s->n_locs++;
        }
    }
    if (obu->frame_end && st->cur >= 0 && !(obu->has_frame_hdr && obu->frame_hdr.show_existing_frame)) {
        const int64_t fi = st->cur;
        const Av1FrameHdr *h = &p->cur;
        for (int i = 0; i < 8; i++)
            if ((h->refresh_frame_flags >> i) & 1)
                st->slot[i] = fi;
        if (s->frames[fi].show_frame && add_display(s, fi) < 0)
            return -1;
        st->cur = -1;
    }
    return 0;
}

int av1fg_scan(FILE *f, int track, int strict, Av1fgScan *s, Av1fgObuCallback cb, void *cb_ctx)
{
    memset(s, 0, sizeof(*s));
#if defined(__linux__) && defined(POSIX_FADV_SEQUENTIAL)
    /* samples are visited in file order: streaming read-ahead beats many small random reads
     * (measured on NVMe: 55 s vs 193 s for a 19.7 GB 4K film) */
    posix_fadvise(fileno(f), 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    if (av1fg_container_open(f, track, &s->cont, s->err, sizeof(s->err)) < 0)
        return -1;
    Av1Parser *p = malloc(sizeof(*p));
    const size_t cap = 1 << 16;
    uint8_t *buf = malloc(cap);
    uint8_t *big = NULL;
    int ret = -1;
    if (!p || !buf)
        goto out;
    av1_parser_init(p);
    p->strict = strict;
    ScanState st;
    for (int i = 0; i < 8; i++)
        st.slot[i] = -1;
    st.cur = -1;
    /* configOBUs (av1C / CodecPrivate): the sequence header */
    if (s->cont.config && s->cont.config_len) {
        size_t o = 0;
        while (o < s->cont.config_len) {
            Av1Obu obu;
            const size_t left = s->cont.config_len - o;
            if (av1_parse_obu(p, s->cont.config + o, left, left, &obu) < 0)
                break;
            if (obu.type == AV1_OBU_SEQUENCE_HEADER) {
                s->seq = p->seq;
                s->have_seq = 1;
            }
            o += obu.total_len;
        }
    }
    const int annexb = s->cont.fmt == AV1FG_FMT_ANNEXB;
    for (size_t u = 0; u < s->cont.n_units; u++) {
        const int64_t uoff = s->cont.units[u].off, uend = uoff + s->cont.units[u].size;
        int64_t off = uoff;
        while (off < uend) {
            const int64_t left = uend - off;
            size_t want = left < 4096 ? (size_t)left : 4096;
            uint8_t *b = buf;
            Av1Obu obu;
            int r;
            for (;;) {
                if (want > cap) {
                    free(big);
                    big = malloc(want);
                    if (!big) {
                        seterr(s, "out of memory");
                        goto out;
                    }
                    b = big;
                }
                const size_t got = av1fg_read_at(f, off, b, want);
                s->bytes_read += (int64_t)got;
                if (got != want) {
                    seterr(s, "read error at %lld", (long long)off);
                    goto out;
                }
                r = av1_parse_obu(p, b, want, (size_t)left, &obu);
                if (r != -2 || want == (size_t)left)
                    break;
                want = want * 8 < (size_t)left ? want * 8 : (size_t)left;
            }
            if (r < 0) {
                seterr(s, "%s sample %zu, offset %lld: %s", av1fg_format_name(s->cont.fmt), u, (long long)off,
                       p->err[0] ? p->err : "malformed OBU");
                goto out;
            }
            if (p->err[0]) {
                s->warnings++;
                p->err[0] = 0;
            }
            if (handle_obu(s, &st, &obu, off, p) < 0) {
                if (!s->err[0])
                    seterr(s, "out of memory");
                goto out;
            }
            if (cb)
                cb(cb_ctx, &obu, off, s);
            if (annexb && (int64_t)obu.total_len != left) {
                seterr(s, "Annex B: obu_size disagrees with obu_length at %lld", (long long)off);
                goto out;
            }
            off += (int64_t)obu.total_len;
        }
    }
    if (!s->have_seq) {
        seterr(s, "no AV1 sequence header found");
        goto out;
    }
    ret = 0;
out:
    free(p);
    free(buf);
    free(big);
    return ret;
}
