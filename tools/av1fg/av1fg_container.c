/*
 * av1fg_container: AV1 sample offsets in IVF, OBU (Section 5), Annex B,
 * MP4/MOV (incl. fragmented) and Matroska/WebM files.
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
#include "av1fg_container.h"
#include "av1hdr.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define fseek64 _fseeki64
#define ftell64 _ftelli64
#else
#include <sys/types.h>
#define fseek64 fseeko
#define ftell64 ftello
#endif

static void seterr(char *err, size_t len, const char *fmt, ...)
{
    if (!err || !len)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, len, fmt, ap);
    va_end(ap);
}

size_t av1fg_read_at(FILE *f, int64_t off, void *buf, size_t n)
{
    if (off < 0 || fseek64(f, off, SEEK_SET))
        return 0;
    return fread(buf, 1, n, f);
}

int64_t av1fg_file_size(FILE *f)
{
    if (fseek64(f, 0, SEEK_END))
        return -1;
    return (int64_t)ftell64(f);
}

const char *av1fg_format_name(Av1fgFormat fmt)
{
    switch (fmt) {
    case AV1FG_FMT_IVF: return "IVF";
    case AV1FG_FMT_OBU: return "OBU (Section 5)";
    case AV1FG_FMT_ANNEXB: return "OBU (Annex B)";
    case AV1FG_FMT_MP4: return "MP4/MOV";
    case AV1FG_FMT_MKV: return "Matroska/WebM";
    default: return "unknown";
    }
}

static int add_unit(Av1fgContainer *c, int64_t off, int64_t size)
{
    if (c->n_units == c->cap_units) {
        const size_t cap = c->cap_units ? c->cap_units * 2 : 4096;
        Av1fgUnit *u = realloc(c->units, cap * sizeof(*u));
        if (!u)
            return -1;
        c->units = u;
        c->cap_units = cap;
    }
    c->units[c->n_units].off = off;
    c->units[c->n_units].size = size;
    c->n_units++;
    return 0;
}

static int add_crc(Av1fgContainer *c, int64_t crc_off, int64_t start, int64_t end, int depth)
{
    if (c->n_crcs == c->cap_crcs) {
        const size_t cap = c->cap_crcs ? c->cap_crcs * 2 : 256;
        Av1fgCrc *u = realloc(c->crcs, cap * sizeof(*u));
        if (!u)
            return -1;
        c->crcs = u;
        c->cap_crcs = cap;
    }
    c->crcs[c->n_crcs].crc_off = crc_off;
    c->crcs[c->n_crcs].start = start;
    c->crcs[c->n_crcs].end = end;
    c->crcs[c->n_crcs].depth = depth;
    c->n_crcs++;
    return 0;
}

void av1fg_container_free(Av1fgContainer *c)
{
    free(c->units);
    free(c->crcs);
    free(c->config);
    memset(c, 0, sizeof(*c));
}

static uint32_t rb32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static uint64_t rb64(const uint8_t *p) { return (uint64_t)rb32(p) << 32 | rb32(p + 4); }
static uint32_t rl32(const uint8_t *p) { return (uint32_t)p[3] << 24 | (uint32_t)p[2] << 16 | (uint32_t)p[1] << 8 | p[0]; }

/* ---------------- IVF ---------------- */

static int open_ivf(FILE *f, Av1fgContainer *c, char *err, size_t errlen)
{
    uint8_t h[32];
    if (av1fg_read_at(f, 0, h, 32) != 32) {
        seterr(err, errlen, "IVF: short header");
        return -1;
    }
    if (memcmp(h + 8, "AV01", 4)) {
        seterr(err, errlen, "IVF: fourcc %.4s is not AV01", (const char *)h + 8);
        return -1;
    }
    int64_t off = h[6] | h[7] << 8;
    while (off + 12 <= c->file_size) {
        uint8_t fh[12];
        if (av1fg_read_at(f, off, fh, 12) != 12)
            break;
        const int64_t sz = rl32(fh);
        if (off + 12 + sz > c->file_size) {
            seterr(err, errlen, "IVF: frame at %lld runs past the end of the file", (long long)off);
            return -1;
        }
        if (add_unit(c, off + 12, sz) < 0)
            return -1;
        off += 12 + sz;
    }
    return 0;
}

/* ---------------- Section 5 / Annex B ---------------- */

static int probe_section5(FILE *f, int64_t size)
{
    uint8_t buf[32];
    int64_t off = 0;
    for (int n = 0; n < 8 && off < size; n++) {
        const size_t got = av1fg_read_at(f, off, buf, sizeof(buf));
        if (got < 2)
            return n > 0;
        if (buf[0] & 0x81) /* forbidden bit, reserved bit */
            return 0;
        if (!(buf[0] & 2)) /* obu_has_size_field */
            return 0;
        const int type = (buf[0] >> 3) & 15;
        if (n == 0 && type != AV1_OBU_TEMPORAL_DELIMITER && type != AV1_OBU_SEQUENCE_HEADER)
            return 0;
        if (type == 0 || (type > 8 && type != 15))
            return 0;
        const int hl = 1 + ((buf[0] >> 2) & 1);
        uint64_t sz;
        const int n2 = av1_leb128(buf + hl, got - hl, &sz);
        if (!n2)
            return 0;
        off += hl + n2 + (int64_t)sz;
        if (off > size)
            return 0;
    }
    return 1;
}

static int read_leb(FILE *f, int64_t off, int64_t end, uint64_t *v, int *len)
{
    uint8_t buf[8];
    size_t n = av1fg_read_at(f, off, buf, 8);
    if (off + (int64_t)n > end)
        n = (size_t)(end - off);
    *len = av1_leb128(buf, n, v);
    return *len ? 0 : -1;
}

/* Annex B: temporal_unit( temporal_unit_size ) { frame_unit_size; frame_unit { obu_length; OBU } } */
static int walk_annexb(FILE *f, Av1fgContainer *c, int probe_only, char *err, size_t errlen)
{
    int64_t off = 0;
    int ntu = 0;
    while (off < c->file_size) {
        uint64_t tu_size, fu_size, obu_len;
        int l;
        if (read_leb(f, off, c->file_size, &tu_size, &l) < 0)
            goto bad;
        off += l;
        const int64_t tu_end = off + (int64_t)tu_size;
        if (tu_end > c->file_size)
            goto bad;
        while (off < tu_end) {
            if (read_leb(f, off, tu_end, &fu_size, &l) < 0)
                goto bad;
            off += l;
            const int64_t fu_end = off + (int64_t)fu_size;
            if (fu_end > tu_end)
                goto bad;
            while (off < fu_end) {
                if (read_leb(f, off, fu_end, &obu_len, &l) < 0)
                    goto bad;
                off += l;
                if (off + (int64_t)obu_len > fu_end || obu_len == 0)
                    goto bad;
                if (probe_only) {
                    uint8_t b0;
                    if (av1fg_read_at(f, off, &b0, 1) != 1 || (b0 & 0x80))
                        goto bad;
                    const int type = (b0 >> 3) & 15;
                    if (type == 0 || (type > 8 && type != 15))
                        goto bad;
                } else if (add_unit(c, off, (int64_t)obu_len) < 0) {
                    return -1;
                }
                off += (int64_t)obu_len;
            }
        }
        if (probe_only && ++ntu >= 4)
            return 0;
    }
    return 0;
bad:
    seterr(err, errlen, "Annex B: bad length field near offset %lld", (long long)off);
    return -1;
}

/* ---------------- MP4 / MOV ---------------- */

typedef struct Mp4Track {
    uint32_t track_id;
    int is_av1, encrypted;
    uint8_t *av1c;
    size_t av1c_len;
    uint32_t sample_size, sample_count;
    uint32_t *sizes;         /* sample_count entries if sample_size == 0 */
    uint64_t *chunk_off;
    uint32_t n_chunks;
    uint32_t (*stsc)[3];     /* first_chunk, samples_per_chunk, sample_description_index */
    uint32_t n_stsc;
    uint32_t trex_default_size;
} Mp4Track;

typedef struct Mp4 {
    FILE *f;
    Mp4Track tr[64];
    int n_tr;
    char *err;
    size_t errlen;
} Mp4;

static int box_header(FILE *f, int64_t off, int64_t end, uint32_t *type, int64_t *hsize, int64_t *bsize)
{
    uint8_t h[16];
    if (off + 8 > end || av1fg_read_at(f, off, h, 16) < 8)
        return -1;
    uint64_t sz = rb32(h);
    *type = rb32(h + 4);
    *hsize = 8;
    if (sz == 1) {
        if (off + 16 > end)
            return -1;
        sz = rb64(h + 8);
        *hsize = 16;
    } else if (sz == 0) {
        sz = (uint64_t)(end - off);
    }
    if (sz < (uint64_t)*hsize || (int64_t)sz > end - off)
        return -1;
    *bsize = (int64_t)sz;
    return 0;
}

#define FOURCC(a, b, c, d) ((uint32_t)(a) << 24 | (uint32_t)(b) << 16 | (uint32_t)(c) << 8 | (uint32_t)(d))

static uint8_t *read_box_payload(FILE *f, int64_t off, int64_t len, size_t max)
{
    if (len < 0 || (uint64_t)len > max)
        return NULL;
    uint8_t *b = malloc(len ? (size_t)len : 1);
    if (!b)
        return NULL;
    if (av1fg_read_at(f, off, b, (size_t)len) != (size_t)len) {
        free(b);
        return NULL;
    }
    return b;
}

static int mp4_stsd(Mp4 *m, Mp4Track *t, int64_t off, int64_t end)
{
    uint8_t h[8];
    if (av1fg_read_at(m->f, off, h, 8) != 8)
        return -1;
    const uint32_t n = rb32(h + 4);
    int64_t p = off + 8;
    for (uint32_t i = 0; i < n && p < end; i++) {
        uint32_t type;
        int64_t hs, bs;
        if (box_header(m->f, p, end, &type, &hs, &bs) < 0)
            return -1;
        if (type == FOURCC('e', 'n', 'c', 'v'))
            t->encrypted = 1;
        if (type == FOURCC('a', 'v', '0', '1') && i == 0) {
            t->is_av1 = 1;
            /* VisualSampleEntry: 78 bytes, then child boxes */
            int64_t q = p + hs + 78;
            const int64_t qend = p + bs;
            while (q + 8 <= qend) {
                uint32_t ct;
                int64_t chs, cbs;
                if (box_header(m->f, q, qend, &ct, &chs, &cbs) < 0)
                    break;
                if (ct == FOURCC('a', 'v', '1', 'C') && cbs - chs >= 4) {
                    free(t->av1c);
                    t->av1c = read_box_payload(m->f, q + chs, cbs - chs, 1 << 20);
                    t->av1c_len = t->av1c ? (size_t)(cbs - chs) : 0;
                }
                q += cbs;
            }
        }
        p += bs;
    }
    return 0;
}

static int mp4_table(Mp4 *m, Mp4Track *t, uint32_t type, int64_t off, int64_t len)
{
    if (len > ((int64_t)1 << 32)) {
        seterr(m->err, m->errlen, "MP4: sample table box too large");
        return -1;
    }
    uint8_t *b = read_box_payload(m->f, off, len, (size_t)len);
    if (!b)
        return -1;
    int ret = -1;
    if (len < 8)
        goto out;
    if (type == FOURCC('s', 't', 's', 'z')) {
        if (len < 12)
            goto out;
        t->sample_size = rb32(b + 4);
        t->sample_count = rb32(b + 8);
        if (!t->sample_size) {
            if ((int64_t)t->sample_count * 4 > len - 12)
                goto out;
            free(t->sizes);
            t->sizes = malloc((size_t)t->sample_count * 4 + 4);
            if (!t->sizes)
                goto out;
            for (uint32_t i = 0; i < t->sample_count; i++)
                t->sizes[i] = rb32(b + 12 + 4 * (size_t)i);
        }
    } else if (type == FOURCC('s', 't', 'z', '2')) {
        if (len < 12)
            goto out;
        const int fs = b[7];
        t->sample_size = 0;
        t->sample_count = rb32(b + 8);
        if (fs != 4 && fs != 8 && fs != 16)
            goto out;
        if (((int64_t)t->sample_count * fs + 7) / 8 > len - 12)
            goto out;
        free(t->sizes);
        t->sizes = malloc((size_t)t->sample_count * 4 + 4);
        if (!t->sizes)
            goto out;
        for (uint32_t i = 0; i < t->sample_count; i++) {
            const uint8_t *q = b + 12;
            if (fs == 16)
                t->sizes[i] = (uint32_t)q[2 * i] << 8 | q[2 * i + 1];
            else if (fs == 8)
                t->sizes[i] = q[i];
            else
                t->sizes[i] = (i & 1) ? (q[i / 2] & 15) : (q[i / 2] >> 4);
        }
    } else if (type == FOURCC('s', 't', 'c', 'o') || type == FOURCC('c', 'o', '6', '4')) {
        const int w = type == FOURCC('c', 'o', '6', '4') ? 8 : 4;
        const uint32_t n = rb32(b + 4);
        if ((int64_t)n * w > len - 8)
            goto out;
        free(t->chunk_off);
        t->chunk_off = malloc((size_t)n * 8 + 8);
        if (!t->chunk_off)
            goto out;
        t->n_chunks = n;
        for (uint32_t i = 0; i < n; i++)
            t->chunk_off[i] = w == 8 ? rb64(b + 8 + 8 * (size_t)i) : rb32(b + 8 + 4 * (size_t)i);
    } else if (type == FOURCC('s', 't', 's', 'c')) {
        const uint32_t n = rb32(b + 4);
        if ((int64_t)n * 12 > len - 8)
            goto out;
        free(t->stsc);
        t->stsc = malloc((size_t)n * 12 + 12);
        if (!t->stsc)
            goto out;
        t->n_stsc = n;
        for (uint32_t i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                t->stsc[i][k] = rb32(b + 8 + 12 * (size_t)i + 4 * k);
    }
    ret = 0;
out:
    free(b);
    if (ret < 0 && !m->err[0])
        seterr(m->err, m->errlen, "MP4: malformed sample table");
    return ret;
}

/* walk containers down to the sample tables */
static int mp4_walk(Mp4 *m, Mp4Track *t, int64_t off, int64_t end, int depth)
{
    while (off + 8 <= end) {
        uint32_t type;
        int64_t hs, bs;
        if (box_header(m->f, off, end, &type, &hs, &bs) < 0) {
            seterr(m->err, m->errlen, "MP4: bad box at %lld", (long long)off);
            return -1;
        }
        const int64_t p = off + hs, len = bs - hs;
        if (type == FOURCC('t', 'r', 'a', 'k') && depth == 0) {
            if (m->n_tr < 64) {
                Mp4Track *nt = &m->tr[m->n_tr++];
                memset(nt, 0, sizeof(*nt));
                if (mp4_walk(m, nt, p, p + len, 1) < 0)
                    return -1;
            }
        } else if (type == FOURCC('m', 'v', 'e', 'x') && depth == 0) {
            if (mp4_walk(m, NULL, p, p + len, 1) < 0)
                return -1;
        } else if (type == FOURCC('t', 'r', 'e', 'x') && !t) {
            uint8_t b[24];
            if (len >= 24 && av1fg_read_at(m->f, p, b, 24) == 24) {
                const uint32_t id = rb32(b + 4);
                for (int i = 0; i < m->n_tr; i++)
                    if (m->tr[i].track_id == id)
                        m->tr[i].trex_default_size = rb32(b + 16);
            }
        } else if (t && (type == FOURCC('m', 'd', 'i', 'a') || type == FOURCC('m', 'i', 'n', 'f') ||
                         type == FOURCC('s', 't', 'b', 'l'))) {
            if (mp4_walk(m, t, p, p + len, depth + 1) < 0)
                return -1;
        } else if (t && type == FOURCC('t', 'k', 'h', 'd')) {
            uint8_t b[32];
            if (len >= 24 && av1fg_read_at(m->f, p, b, 32) >= 24)
                t->track_id = b[0] == 1 ? rb32(b + 20) : rb32(b + 12);
        } else if (t && type == FOURCC('s', 't', 's', 'd')) {
            if (mp4_stsd(m, t, p, p + len) < 0)
                return -1;
        } else if (t && (type == FOURCC('s', 't', 's', 'z') || type == FOURCC('s', 't', 'z', '2') ||
                         type == FOURCC('s', 't', 'c', 'o') || type == FOURCC('c', 'o', '6', '4') ||
                         type == FOURCC('s', 't', 's', 'c'))) {
            if (mp4_table(m, t, type, p, len) < 0)
                return -1;
        }
        off += bs;
    }
    return 0;
}

static int mp4_moof(Mp4 *m, Av1fgContainer *c, Mp4Track *t, int64_t moof_off, int64_t end)
{
    /* moof children: mfhd, traf* */
    int64_t off = moof_off;
    uint32_t type;
    int64_t hs, bs;
    if (box_header(m->f, off, end, &type, &hs, &bs) < 0)
        return -1;
    const int64_t moof_end = off + bs;
    int64_t prev_traf_data_end = moof_off; /* base for a traf without base_data_offset/default-base-is-moof */
    off += hs;
    while (off + 8 <= moof_end) {
        if (box_header(m->f, off, moof_end, &type, &hs, &bs) < 0)
            return -1;
        if (type == FOURCC('t', 'r', 'a', 'f')) {
            int64_t q = off + hs;
            const int64_t traf_end = off + bs;
            int64_t base = -1;
            uint32_t default_size = t->trex_default_size, id = 0;
            int ours = 0;
            int64_t data_end = prev_traf_data_end;
            int64_t next = -1; /* where the next trun without data_offset starts */
            while (q + 8 <= traf_end) {
                uint32_t ct;
                int64_t chs, cbs;
                if (box_header(m->f, q, traf_end, &ct, &chs, &cbs) < 0)
                    return -1;
                const int64_t clen = cbs - chs;
                if (ct == FOURCC('t', 'f', 'h', 'd') && clen >= 8) {
                    uint8_t b[40];
                    const size_t n = av1fg_read_at(m->f, q + chs, b, clen < 40 ? (size_t)clen : 40);
                    const uint32_t flags = (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
                    id = rb32(b + 4);
                    ours = id == t->track_id;
                    size_t p = 8;
                    if (flags & 1) {
                        if (p + 8 > n)
                            return -1;
                        base = (int64_t)rb64(b + p);
                        p += 8;
                    } else if (flags & 0x20000) {
                        base = moof_off;
                    } else {
                        base = prev_traf_data_end;
                    }
                    if (flags & 2)
                        p += 4;
                    if (flags & 8)
                        p += 4;
                    if (flags & 0x10) {
                        if (p + 4 > n)
                            return -1;
                        default_size = rb32(b + p);
                    }
                    next = base;
                } else if (ct == FOURCC('t', 'r', 'u', 'n') && clen >= 8) {
                    if (base < 0)
                        return -1;
                    uint8_t *b = read_box_payload(m->f, q + chs, clen, (size_t)1 << 28);
                    if (!b)
                        return -1;
                    const uint32_t flags = (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
                    const uint32_t count = rb32(b + 4);
                    size_t p = 8;
                    int64_t data = next;
                    if (flags & 1) {
                        data = base + (int32_t)rb32(b + p);
                        p += 4;
                    }
                    if (flags & 4)
                        p += 4;
                    const int per = !!(flags & 0x100) + !!(flags & 0x200) + !!(flags & 0x400) + !!(flags & 0x800);
                    if (p + (uint64_t)count * per * 4 > (uint64_t)clen) {
                        free(b);
                        return -1;
                    }
                    for (uint32_t i = 0; i < count; i++) {
                        if (flags & 0x100)
                            p += 4;
                        uint32_t sz = default_size;
                        if (flags & 0x200) {
                            sz = rb32(b + p);
                            p += 4;
                        }
                        if (flags & 0x400)
                            p += 4;
                        if (flags & 0x800)
                            p += 4;
                        if (ours) {
                            if (data + sz > c->file_size) {
                                free(b);
                                return -1;
                            }
                            if (add_unit(c, data, sz) < 0) {
                                free(b);
                                return -1;
                            }
                        }
                        data += sz;
                    }
                    free(b);
                    next = data;
                    data_end = data;
                }
                q += cbs;
            }
            prev_traf_data_end = data_end;
        }
        off += bs;
    }
    return 0;
}

static int open_mp4(FILE *f, int track, Av1fgContainer *c, char *err, size_t errlen)
{
    Mp4 *m = calloc(1, sizeof(*m));
    if (!m)
        return -1;
    m->f = f;
    m->err = err;
    m->errlen = errlen;
    err[0] = 0;
    int ret = -1;
    /* top level: find moov */
    int64_t off = 0;
    int have_moov = 0;
    while (off + 8 <= c->file_size) {
        uint32_t type;
        int64_t hs, bs;
        if (box_header(f, off, c->file_size, &type, &hs, &bs) < 0) {
            seterr(err, errlen, "MP4: bad top-level box at %lld", (long long)off);
            goto out;
        }
        if (type == FOURCC('m', 'o', 'o', 'v')) {
            if (mp4_walk(m, NULL, off + hs, off + bs, 0) < 0)
                goto out;
            have_moov = 1;
        }
        off += bs;
    }
    if (!have_moov) {
        seterr(err, errlen, "MP4: no moov box");
        goto out;
    }
    Mp4Track *t = NULL;
    for (int i = 0; i < m->n_tr; i++) {
        if (m->tr[i].is_av1 && (!track || (uint32_t)track == m->tr[i].track_id)) {
            if (!t)
                t = &m->tr[i];
        } else {
            c->other_tracks++;
        }
    }
    if (!t) {
        seterr(err, errlen, "MP4: no AV1 ('av01') track%s", track ? " with that track_ID" : "");
        goto out;
    }
    if (t->encrypted) {
        seterr(err, errlen, "MP4: the AV1 track is encrypted");
        goto out;
    }
    c->track_id = (int)t->track_id;
    if (t->av1c && t->av1c_len > 4) {
        c->config_len = t->av1c_len - 4;
        c->config = malloc(c->config_len);
        if (c->config)
            memcpy(c->config, t->av1c + 4, c->config_len);
    }
    /* regular sample table */
    if (t->sample_count) {
        if (!t->chunk_off || !t->stsc || !t->n_stsc) {
            seterr(err, errlen, "MP4: missing stco/co64 or stsc");
            goto out;
        }
        uint32_t s = 0;
        for (uint32_t e = 0; e < t->n_stsc && s < t->sample_count; e++) {
            const uint32_t first = t->stsc[e][0];
            const uint32_t last = e + 1 < t->n_stsc ? t->stsc[e + 1][0] - 1 : t->n_chunks;
            const uint32_t per = t->stsc[e][1];
            if (first < 1 || last > t->n_chunks + 0u) {
                seterr(err, errlen, "MP4: bad stsc");
                goto out;
            }
            for (uint32_t ch = first; ch <= last && s < t->sample_count; ch++) {
                int64_t o = (int64_t)t->chunk_off[ch - 1];
                for (uint32_t k = 0; k < per && s < t->sample_count; k++, s++) {
                    const uint32_t sz = t->sample_size ? t->sample_size : t->sizes[s];
                    if (o + sz > c->file_size) {
                        seterr(err, errlen, "MP4: sample %u runs past the end of the file", s);
                        goto out;
                    }
                    if (add_unit(c, o, sz) < 0)
                        goto out;
                    o += sz;
                }
            }
        }
        if (s != t->sample_count) {
            seterr(err, errlen, "MP4: sample table covers %u of %u samples", s, t->sample_count);
            goto out;
        }
    }
    /* fragments */
    off = 0;
    int nfrag = 0;
    while (off + 8 <= c->file_size) {
        uint32_t type;
        int64_t hs, bs;
        if (box_header(f, off, c->file_size, &type, &hs, &bs) < 0)
            break;
        if (type == FOURCC('m', 'o', 'o', 'f')) {
            if (mp4_moof(m, c, t, off, c->file_size) < 0) {
                seterr(err, errlen, "MP4: bad movie fragment at %lld", (long long)off);
                goto out;
            }
            nfrag++;
        }
        off += bs;
    }
    snprintf(c->info, sizeof(c->info), "track_ID %u%s", t->track_id, nfrag ? ", fragmented" : "");
    ret = 0;
out:
    for (int i = 0; i < m->n_tr; i++) {
        free(m->tr[i].av1c);
        free(m->tr[i].sizes);
        free(m->tr[i].chunk_off);
        free(m->tr[i].stsc);
    }
    free(m);
    return ret;
}

/* ---------------- Matroska / WebM ---------------- */

#define MKV_EBML 0x1A45DFA3u
#define MKV_SEGMENT 0x18538067u
#define MKV_CLUSTER 0x1F43B675u
#define MKV_TRACKS 0x1654AE6Bu
#define MKV_CRC32 0xBFu
#define MKV_VOID 0xECu

static int ebml_id(const uint8_t *b, size_t n, uint32_t *id)
{
    if (!n)
        return 0;
    int len = 0;
    for (int i = 0; i < 4; i++)
        if (b[0] & (0x80 >> i)) {
            len = i + 1;
            break;
        }
    if (!len || (size_t)len > n)
        return 0;
    uint32_t v = 0;
    for (int i = 0; i < len; i++)
        v = v << 8 | b[i];
    *id = v;
    return len;
}

/* returns length; *unknown set for an all-ones size */
static int ebml_size(const uint8_t *b, size_t n, uint64_t *size, int *unknown)
{
    if (!n)
        return 0;
    int len = 0;
    for (int i = 0; i < 8; i++)
        if (b[0] & (0x80 >> i)) {
            len = i + 1;
            break;
        }
    if (!len || (size_t)len > n)
        return 0;
    uint64_t v = b[0] & (0xff >> len);
    int ones = v == (uint64_t)(0xff >> len);
    for (int i = 1; i < len; i++) {
        v = v << 8 | b[i];
        ones &= b[i] == 0xff;
    }
    *size = v;
    *unknown = ones;
    return len;
}

typedef struct EbmlEl {
    uint32_t id;
    int64_t off, data, size; /* size -1 = unknown */
    int64_t end;
} EbmlEl;

static int ebml_read(FILE *f, int64_t off, int64_t end, EbmlEl *e)
{
    uint8_t b[12];
    size_t n = av1fg_read_at(f, off, b, 12);
    if (off + (int64_t)n > end)
        n = (size_t)(end - off);
    const int il = ebml_id(b, n, &e->id);
    if (!il)
        return -1;
    uint64_t sz;
    int unk;
    const int sl = ebml_size(b + il, n - il, &sz, &unk);
    if (!sl)
        return -1;
    e->off = off;
    e->data = off + il + sl;
    e->size = unk ? -1 : (int64_t)sz;
    e->end = unk ? end : e->data + (int64_t)sz;
    if (e->end > end)
        return -1;
    return 0;
}

static int is_level1(uint32_t id)
{
    return id == MKV_CLUSTER || id == MKV_TRACKS || id == 0x114D9B74u || id == 0x1549A966u ||
           id == 0x1C53BB6Bu || id == 0x1043A770u || id == 0x1254C367u || id == 0x1941A469u ||
           id == MKV_SEGMENT || id == MKV_EBML;
}

typedef struct Mkv {
    FILE *f;
    Av1fgContainer *c;
    int want_track;      /* requested track or 0 */
    int track;           /* chosen AV1 TrackNumber, 0 if not known yet */
    int content_encoding;
    char *err;
    size_t errlen;
} Mkv;

static int64_t read_uint_el(FILE *f, const EbmlEl *e)
{
    uint8_t b[8];
    if (e->size < 1 || e->size > 8 || av1fg_read_at(f, e->data, b, (size_t)e->size) != (size_t)e->size)
        return -1;
    int64_t v = 0;
    for (int i = 0; i < e->size; i++)
        v = v << 8 | b[i];
    return v;
}

static int mkv_tracks(Mkv *m, const EbmlEl *tracks)
{
    int64_t off = tracks->data;
    while (off < tracks->end) {
        EbmlEl te;
        if (ebml_read(m->f, off, tracks->end, &te) < 0 || te.size < 0)
            return -1;
        if (te.id == 0xAE) { /* TrackEntry */
            int64_t num = 0;
            int is_av1 = 0, enc = 0;
            uint8_t *priv = NULL;
            size_t priv_len = 0;
            int64_t q = te.data;
            while (q < te.end) {
                EbmlEl ch;
                if (ebml_read(m->f, q, te.end, &ch) < 0 || ch.size < 0) {
                    free(priv);
                    return -1;
                }
                if (ch.id == 0xD7) {
                    num = read_uint_el(m->f, &ch);
                } else if (ch.id == 0x86 && ch.size < 64) {
                    char id[64] = { 0 };
                    av1fg_read_at(m->f, ch.data, id, (size_t)ch.size);
                    is_av1 = !strcmp(id, "V_AV1");
                } else if (ch.id == 0x63A2 && ch.size < (1 << 20)) {
                    free(priv);
                    priv = read_box_payload(m->f, ch.data, ch.size, 1 << 20);
                    priv_len = priv ? (size_t)ch.size : 0;
                } else if (ch.id == 0x6D80) {
                    enc = 1;
                }
                q = ch.end;
            }
            if (is_av1 && !m->track && (!m->want_track || m->want_track == num)) {
                m->track = (int)num;
                m->content_encoding = enc;
                if (priv && priv_len > 4) {
                    m->c->config_len = priv_len - 4;
                    m->c->config = malloc(m->c->config_len);
                    if (m->c->config)
                        memcpy(m->c->config, priv + 4, m->c->config_len);
                }
            } else {
                m->c->other_tracks++;
            }
            free(priv);
        }
        off = te.end;
    }
    return 0;
}

static int mkv_block(Mkv *m, const EbmlEl *b)
{
    uint8_t h[12];
    const size_t n = av1fg_read_at(m->f, b->data, h, 12);
    uint64_t tn;
    int unk;
    const int tl = ebml_size(h, n, &tn, &unk);
    if (!tl || (size_t)tl + 3 > n || b->size < tl + 3)
        return -1;
    if ((int)tn != m->track)
        return 0;
    const int flags = h[tl + 2];
    if ((flags >> 1) & 3) {
        seterr(m->err, m->errlen, "Matroska: laced AV1 block at %lld (lacing is not allowed for AV1)",
               (long long)b->off);
        return -2;
    }
    return add_unit(m->c, b->data + tl + 3, b->size - tl - 3);
}

/* Walk a master element's children; records CRC-32 elements; returns the end offset. */
static int64_t mkv_cluster(Mkv *m, const EbmlEl *cl, int depth)
{
    int64_t off = cl->data;
    int first = 1;
    while (off < cl->end) {
        EbmlEl e;
        if (ebml_read(m->f, off, cl->end, &e) < 0)
            return cl->size < 0 ? off : -1; /* unknown-size cluster ends at garbage / EOF */
        if (cl->size < 0 && is_level1(e.id))
            return off;
        if (e.size < 0)
            return -1;
        if (e.id == MKV_CRC32 && first && e.size == 4) {
            if (cl->size < 0) {
                seterr(m->err, m->errlen, "Matroska: CRC-32 in an unknown-size element");
                return -1;
            }
            if (add_crc(m->c, e.data, e.end, cl->end, depth) < 0)
                return -1;
        } else if (e.id == 0xA3 || e.id == 0xA1) { /* SimpleBlock, Block */
            const int r = mkv_block(m, &e);
            if (r < 0)
                return r == -2 ? -2 : -1;
        } else if (e.id == 0xA0) { /* BlockGroup */
            if (mkv_cluster(m, &e, depth + 1) < 0)
                return -1;
        } else if (e.id == 0xAF) { /* EncryptedBlock */
            seterr(m->err, m->errlen, "Matroska: EncryptedBlock");
            return -1;
        }
        first = 0;
        off = e.end;
    }
    return cl->end;
}

static int open_mkv(FILE *f, int track, Av1fgContainer *c, char *err, size_t errlen)
{
    Mkv m = { f, c, track, 0, 0, err, errlen };
    EbmlEl e;
    if (ebml_read(f, 0, c->file_size, &e) < 0 || e.id != MKV_EBML || e.size < 0) {
        seterr(err, errlen, "Matroska: bad EBML header");
        return -1;
    }
    int64_t off = e.end;
    while (off < c->file_size) {
        if (ebml_read(f, off, c->file_size, &e) < 0) {
            seterr(err, errlen, "Matroska: bad element at %lld", (long long)off);
            return -1;
        }
        if (e.id == MKV_SEGMENT)
            break;
        off = e.end;
    }
    if (e.id != MKV_SEGMENT) {
        seterr(err, errlen, "Matroska: no Segment");
        return -1;
    }
    const EbmlEl seg = e;
    /* pass 1: Tracks (usually before the first Cluster) */
    off = seg.data;
    int64_t first_cluster = -1;
    while (off < seg.end) {
        if (ebml_read(f, off, seg.end, &e) < 0)
            break;
        if (e.id == MKV_CRC32 && off == seg.data && e.size == 4)
            add_crc(c, e.data, e.end, seg.end, 0);
        if (e.id == MKV_TRACKS) {
            if (e.size < 0 || mkv_tracks(&m, &e) < 0) {
                seterr(err, errlen, "Matroska: bad Tracks");
                return -1;
            }
            break;
        }
        if (e.id == MKV_CLUSTER && first_cluster < 0)
            first_cluster = off;
        if (e.size < 0) {
            /* an unknown-size cluster before Tracks: skip by walking it */
            const int t = m.track;
            m.track = -1;
            const int64_t end = mkv_cluster(&m, &e, 1);
            m.track = t;
            if (end < 0)
                return -1;
            off = end;
            continue;
        }
        off = e.end;
    }
    if (!m.track) {
        seterr(err, errlen, "Matroska: no V_AV1 track%s", track ? " with that number" : "");
        return -1;
    }
    if (m.content_encoding) {
        seterr(err, errlen, "Matroska: the AV1 track uses ContentEncoding (compression or encryption)");
        return -1;
    }
    c->track_id = m.track;
    /* pass 2: clusters */
    off = seg.data;
    int nclusters = 0;
    while (off < seg.end) {
        if (ebml_read(f, off, seg.end, &e) < 0) {
            if (seg.size < 0)
                break;
            seterr(err, errlen, "Matroska: bad element at %lld", (long long)off);
            return -1;
        }
        if (e.id == MKV_CLUSTER) {
            const int64_t end = mkv_cluster(&m, &e, 1);
            if (end < 0) {
                if (!err[0])
                    seterr(err, errlen, "Matroska: bad Cluster at %lld", (long long)off);
                return -1;
            }
            nclusters++;
            off = end;
            continue;
        }
        if (e.size < 0) {
            seterr(err, errlen, "Matroska: unknown-size element 0x%X", e.id);
            return -1;
        }
        off = e.end;
    }
    snprintf(c->info, sizeof(c->info), "track %d, %d clusters%s", m.track, nclusters,
             c->n_crcs ? ", CRC-32 elements" : "");
    return 0;
}

int av1fg_container_open(FILE *f, int track, Av1fgContainer *c, char *err, size_t errlen)
{
    memset(c, 0, sizeof(*c));
    err[0] = 0;
    c->file_size = av1fg_file_size(f);
    if (c->file_size < 0) {
        seterr(err, errlen, "cannot seek in the input");
        return -1;
    }
    uint8_t h[16] = { 0 };
    av1fg_read_at(f, 0, h, 16);
    int r;
    if (!memcmp(h, "DKIF", 4)) {
        c->fmt = AV1FG_FMT_IVF;
        r = open_ivf(f, c, err, errlen);
    } else if (rb32(h) == MKV_EBML) {
        c->fmt = AV1FG_FMT_MKV;
        r = open_mkv(f, track, c, err, errlen);
    } else if (!memcmp(h + 4, "ftyp", 4) || !memcmp(h + 4, "moov", 4) || !memcmp(h + 4, "mdat", 4) ||
               !memcmp(h + 4, "free", 4) || !memcmp(h + 4, "wide", 4) || !memcmp(h + 4, "styp", 4)) {
        c->fmt = AV1FG_FMT_MP4;
        r = open_mp4(f, track, c, err, errlen);
    } else if (probe_section5(f, c->file_size)) {
        c->fmt = AV1FG_FMT_OBU;
        r = add_unit(c, 0, c->file_size);
    } else {
        Av1fgContainer probe = *c;
        if (walk_annexb(f, &probe, 1, err, errlen) == 0) {
            c->fmt = AV1FG_FMT_ANNEXB;
            r = walk_annexb(f, c, 0, err, errlen);
        } else {
            seterr(err, errlen, "unknown format (not IVF, MP4/MOV, Matroska/WebM, or an AV1 OBU stream)");
            r = -1;
        }
    }
    if (r < 0 && !err[0])
        seterr(err, errlen, "%s: cannot parse (out of memory or malformed)", av1fg_format_name(c->fmt));
    return r < 0 ? -1 : 0;
}

/* ---------------- CRC-32 elements (IEEE CRC-32, as zlib / EBML) ---------------- */

static int cmp_edit(const void *a, const void *b)
{
    const Av1fgEdit *x = a, *y = b;
    return x->off < y->off ? -1 : x->off > y->off;
}

static uint32_t gf2_times(const uint32_t *mat, uint32_t vec)
{
    uint32_t sum = 0;
    while (vec) {
        if (vec & 1)
            sum ^= *mat;
        vec >>= 1;
        mat++;
    }
    return sum;
}

static void gf2_square(uint32_t *sq, const uint32_t *mat)
{
    for (int n = 0; n < 32; n++)
        sq[n] = gf2_times(mat, mat[n]);
}

/* the CRC register after len more zero bytes */
static uint32_t crc_shift(uint32_t crc, uint64_t len)
{
    static uint32_t op[64][32];
    static int init;
    if (!init) {
        uint32_t odd[32], even[32], row = 1;
        odd[0] = 0xEDB88320u; /* one zero bit */
        for (int n = 1; n < 32; n++) {
            odd[n] = row;
            row <<= 1;
        }
        gf2_square(even, odd); /* 2 bits */
        gf2_square(odd, even); /* 4 bits */
        gf2_square(op[0], odd); /* 8 bits: one byte */
        for (int k = 1; k < 64; k++)
            gf2_square(op[k], op[k - 1]);
        init = 1;
    }
    for (int k = 0; len; k++, len >>= 1)
        if (len & 1)
            crc = gf2_times(op[k], crc);
    return crc;
}

int av1fg_crc_edits(FILE *f, const Av1fgContainer *c, Av1fgEdit **edits, size_t *n, size_t *cap)
{
    qsort(*edits, *n, sizeof(**edits), cmp_edit);
    if (!c->n_crcs || !*n)
        return 0;
    uint32_t table[256];
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t v = i;
        for (int k = 0; k < 8; k++)
            v = v & 1 ? 0xEDB88320u ^ (v >> 1) : v >> 1;
        table[i] = v;
    }
    int maxd = 0, changed = 0;
    for (size_t i = 0; i < c->n_crcs; i++)
        if (c->crcs[i].depth > maxd)
            maxd = c->crcs[i].depth;
    for (int d = maxd; d >= 0; d--) {
        const size_t n0 = *n;
        for (size_t i = 0; i < c->n_crcs; i++) {
            const Av1fgCrc *k = &c->crcs[i];
            if (k->depth != d)
                continue;
            /* first edit at or after k->start */
            size_t lo = 0, hi = n0;
            while (lo < hi) {
                const size_t mid = (lo + hi) / 2;
                if ((*edits)[mid].off < k->start)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo >= n0 || (*edits)[lo].off >= k->end)
                continue;
            uint32_t delta = 0;
            for (size_t j = lo; j < n0 && (*edits)[j].off < k->end;) {
                uint32_t raw = 0;
                int64_t last = (*edits)[j].off - 1;
                while (j < n0 && (*edits)[j].off == last + 1 && (*edits)[j].off < k->end) {
                    raw = (raw >> 8) ^ table[(raw ^ ((*edits)[j].old ^ (*edits)[j].val)) & 0xff];
                    last = (*edits)[j].off;
                    j++;
                }
                delta ^= crc_shift(raw, (uint64_t)(k->end - last - 1));
            }
            uint8_t b[4];
            if (av1fg_read_at(f, k->crc_off, b, 4) != 4)
                return -1;
            const uint32_t old = rl32(b), nw = old ^ delta;
            if (old == nw)
                continue;
            for (int q = 0; q < 4; q++) {
                const uint8_t ob = (uint8_t)(old >> (8 * q)), nb = (uint8_t)(nw >> (8 * q));
                if (ob == nb)
                    continue;
                if (*n == *cap) {
                    const size_t nc = *cap ? *cap * 2 : 1024;
                    Av1fgEdit *e = realloc(*edits, nc * sizeof(*e));
                    if (!e)
                        return -1;
                    *edits = e;
                    *cap = nc;
                }
                (*edits)[(*n)++] = (Av1fgEdit){ k->crc_off + q, ob, nb };
            }
            changed++;
        }
        if (*n != n0)
            qsort(*edits, *n, sizeof(**edits), cmp_edit);
    }
    return changed;
}

