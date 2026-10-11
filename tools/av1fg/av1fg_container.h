/*
 * av1fg_container: find the AV1 samples (temporal units / OBUs) in IVF,
 * low-overhead OBU streams (Section 5), Annex B, MP4/MOV (also fragmented)
 * and Matroska/WebM files, without copying them. Byte offsets only; the
 * caller reads and patches the samples in place.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#ifndef AV1FG_CONTAINER_H
#define AV1FG_CONTAINER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    AV1FG_FMT_UNKNOWN = 0,
    AV1FG_FMT_IVF,
    AV1FG_FMT_OBU,      /* low-overhead bitstream format (Section 5) */
    AV1FG_FMT_ANNEXB,   /* length-delimited bitstream format (Annex B) */
    AV1FG_FMT_MP4,
    AV1FG_FMT_MKV,
} Av1fgFormat;

typedef struct Av1fgUnit {
    int64_t off;        /* file offset of the first OBU */
    int64_t size;       /* bytes: one sample (IVF frame, MP4 sample, Matroska block
                           frame), the whole file (Section 5) or one OBU (Annex B) */
} Av1fgUnit;

/* An EBML CRC-32 element and the bytes it covers (Matroska). */
typedef struct Av1fgCrc {
    int64_t crc_off;    /* offset of the 4 CRC bytes */
    int64_t start, end; /* covered range [start, end) */
    int depth;          /* nesting depth: inner CRCs are recomputed first */
} Av1fgCrc;

typedef struct Av1fgContainer {
    Av1fgFormat fmt;
    int64_t file_size;
    int track_id;           /* MP4 track_ID / Matroska TrackNumber of the AV1 track */
    uint8_t *config;        /* av1C configOBUs (MP4 / Matroska CodecPrivate), may be NULL */
    size_t config_len;
    Av1fgUnit *units;
    size_t n_units, cap_units;
    Av1fgCrc *crcs;
    size_t n_crcs, cap_crcs;
    int other_tracks;       /* number of non-AV1 tracks (kept untouched) */
    char info[256];         /* human-readable notes */
} Av1fgContainer;

/* Portable 64-bit positioned read. Returns the number of bytes read. */
size_t av1fg_read_at(FILE *f, int64_t off, void *buf, size_t n);
int64_t av1fg_file_size(FILE *f);

const char *av1fg_format_name(Av1fgFormat fmt);

/* Detect the format and list the AV1 samples in decode order.
 * track: Matroska TrackNumber / MP4 track_ID to use, or 0 for the first AV1 track.
 * Returns 0, or -1 with a message in err. */
int av1fg_container_open(FILE *f, int track, Av1fgContainer *c, char *err, size_t errlen);
void av1fg_container_free(Av1fgContainer *c);

/* One byte of the file that changes. */
typedef struct Av1fgEdit {
    int64_t off;
    uint8_t old, val;
} Av1fgEdit;

/* Add the edits of the CRC-32 elements that cover edited bytes. CRC-32 is
 * linear, so the new values follow from the old CRC and the byte changes
 * alone, without reading the covered data (zero-byte shift operators as in
 * zlib's crc32_combine); nested CRCs are updated innermost first. *edits is
 * sorted by offset on return. f: the input. Returns the number of CRC-32
 * elements changed, or -1. */
int av1fg_crc_edits(FILE *f, const Av1fgContainer *c, Av1fgEdit **edits, size_t *n, size_t *cap);

#endif
