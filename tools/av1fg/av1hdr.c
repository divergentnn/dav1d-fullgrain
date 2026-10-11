/*
 * av1hdr: a small, standalone AV1 header parser (no pixel decoding).
 * Section numbers refer to the AV1 Bitstream & Decoding Process
 * Specification (with errata).
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#include "av1hdr.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define NUM_REF_FRAMES 8
#define REFS_PER_FRAME 7
#define PRIMARY_REF_NONE 7
#define SELECT_SCREEN_CONTENT_TOOLS 2
#define SELECT_INTEGER_MV 2
#define WARPEDMODEL_PREC_BITS 16
#define GM_ABS_TRANS_BITS 12
#define GM_ABS_TRANS_ONLY_BITS 9
#define GM_ABS_ALPHA_BITS 12
#define GM_ALPHA_PREC_BITS 15
#define GM_TRANS_PREC_BITS 6
#define GM_TRANS_ONLY_PREC_BITS 3
#define MAX_TILE_WIDTH 4096
#define MAX_TILE_AREA (4096 * 2304)
#define MAX_TILE_ROWS 64
#define MAX_TILE_COLS 64

enum { GM_IDENTITY = 0, GM_TRANSLATION = 1, GM_ROTZOOM = 2, GM_AFFINE = 3 };

typedef struct BitReader {
    const uint8_t *buf;
    int64_t nbits;      /* bits available in buf */
    int64_t full_bits;  /* bits in the whole payload */
    int64_t pos;
    int overrun;        /* read past the available bytes */
} BitReader;

static void br_init(BitReader *b, const uint8_t *buf, size_t avail, size_t full)
{
    b->buf = buf;
    b->nbits = (int64_t)(avail < full ? avail : full) * 8;
    b->full_bits = (int64_t)full * 8;
    b->pos = 0;
    b->overrun = 0;
}

static unsigned f(BitReader *b, int n)
{
    unsigned v = 0;
    for (int i = 0; i < n; i++) {
        if (b->pos >= b->nbits) {
            b->overrun = 1;
            b->pos++;
            v <<= 1;
            continue;
        }
        v = (v << 1) | ((b->buf[b->pos >> 3] >> (7 - (b->pos & 7))) & 1);
        b->pos++;
    }
    return v;
}

static int su(BitReader *b, int n)
{
    int v = (int)f(b, n);
    const int sign = 1 << (n - 1);
    if (v & sign)
        v -= 2 * sign;
    return v;
}

static unsigned ns(BitReader *b, unsigned n)
{
    int w = 0;
    for (unsigned x = n; x; x >>= 1)
        w++;
    const unsigned m = (1u << w) - n;
    const unsigned v = f(b, w - 1);
    if (v < m)
        return v;
    const unsigned extra = f(b, 1);
    return (v << 1) - m + extra;
}

static uint32_t uvlc(BitReader *b)
{
    int lz = 0;
    while (!b->overrun) {
        if (f(b, 1))
            break;
        lz++;
        if (lz > 40)
            break;
    }
    if (lz >= 32)
        return 0xffffffffu;
    return f(b, lz) + ((1u << lz) - 1);
}

static void byte_alignment(BitReader *b)
{
    while (b->pos & 7)
        f(b, 1);
}

static void seterr(char *err, size_t len, const char *fmt, ...)
{
    if (!err || !len)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, len, fmt, ap);
    va_end(ap);
}

int av1_leb128(const uint8_t *buf, size_t len, uint64_t *value)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        if ((size_t)i >= len)
            return 0;
        v |= (uint64_t)(buf[i] & 0x7f) << (i * 7);
        if (!(buf[i] & 0x80)) {
            *value = v;
            return i + 1;
        }
    }
    return 0;
}

void av1_put_bits16(uint8_t *buf, int64_t bitpos, unsigned value)
{
    for (int i = 0; i < 16; i++) {
        const int64_t p = bitpos + i;
        const int bit = (value >> (15 - i)) & 1;
        const uint8_t mask = (uint8_t)(0x80 >> (p & 7));
        if (bit)
            buf[p >> 3] |= mask;
        else
            buf[p >> 3] &= (uint8_t)~mask;
    }
}

unsigned av1_get_bits16(const uint8_t *buf, int64_t bitpos)
{
    unsigned v = 0;
    for (int i = 0; i < 16; i++) {
        const int64_t p = bitpos + i;
        v = (v << 1) | ((buf[p >> 3] >> (7 - (p & 7))) & 1);
    }
    return v;
}

/* ---- sequence header (5.5) ---- */

int av1_parse_sequence_header(const uint8_t *buf, size_t len, Av1SeqHdr *s, char *err, size_t errlen)
{
    BitReader b;
    br_init(&b, buf, len, len);
    memset(s, 0, sizeof(*s));
    s->seq_profile = f(&b, 3);
    s->still_picture = f(&b, 1);
    s->reduced_still_picture_header = f(&b, 1);
    if (s->seq_profile > 2) {
        seterr(err, errlen, "sequence header: seq_profile %d", s->seq_profile);
        return -1;
    }
    if (s->reduced_still_picture_header) {
        s->seq_level_idx[0] = f(&b, 5);
    } else {
        s->timing_info_present_flag = f(&b, 1);
        if (s->timing_info_present_flag) {
            s->num_units_in_display_tick = f(&b, 32);
            s->time_scale = f(&b, 32);
            s->equal_picture_interval = f(&b, 1);
            if (s->equal_picture_interval)
                s->num_ticks_per_picture_minus_1 = uvlc(&b);
            s->decoder_model_info_present_flag = f(&b, 1);
            if (s->decoder_model_info_present_flag) {
                s->buffer_delay_length_minus_1 = f(&b, 5);
                f(&b, 32); /* num_units_in_decoding_tick */
                s->buffer_removal_time_length_minus_1 = f(&b, 5);
                s->frame_presentation_time_length_minus_1 = f(&b, 5);
            }
        }
        s->initial_display_delay_present_flag = f(&b, 1);
        s->operating_points_cnt_minus_1 = f(&b, 5);
        for (int i = 0; i <= s->operating_points_cnt_minus_1; i++) {
            s->operating_point_idc[i] = f(&b, 12);
            s->seq_level_idx[i] = f(&b, 5);
            s->seq_tier[i] = s->seq_level_idx[i] > 7 ? (int)f(&b, 1) : 0;
            if (s->decoder_model_info_present_flag) {
                s->decoder_model_present_for_this_op[i] = f(&b, 1);
                if (s->decoder_model_present_for_this_op[i]) {
                    const int n = s->buffer_delay_length_minus_1 + 1;
                    f(&b, n); /* decoder_buffer_delay */
                    f(&b, n); /* encoder_buffer_delay */
                    f(&b, 1); /* low_delay_mode_flag */
                }
            }
            if (s->initial_display_delay_present_flag) {
                if (f(&b, 1))
                    f(&b, 4);
            }
        }
    }
    s->frame_width_bits_minus_1 = f(&b, 4);
    s->frame_height_bits_minus_1 = f(&b, 4);
    s->max_frame_width_minus_1 = f(&b, s->frame_width_bits_minus_1 + 1);
    s->max_frame_height_minus_1 = f(&b, s->frame_height_bits_minus_1 + 1);
    if (!s->reduced_still_picture_header)
        s->frame_id_numbers_present_flag = f(&b, 1);
    if (s->frame_id_numbers_present_flag) {
        s->delta_frame_id_length_minus_2 = f(&b, 4);
        s->additional_frame_id_length_minus_1 = f(&b, 3);
    }
    s->use_128x128_superblock = f(&b, 1);
    s->enable_filter_intra = f(&b, 1);
    s->enable_intra_edge_filter = f(&b, 1);
    if (s->reduced_still_picture_header) {
        s->seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        s->seq_force_integer_mv = SELECT_INTEGER_MV;
    } else {
        s->enable_interintra_compound = f(&b, 1);
        s->enable_masked_compound = f(&b, 1);
        s->enable_warped_motion = f(&b, 1);
        s->enable_dual_filter = f(&b, 1);
        s->enable_order_hint = f(&b, 1);
        if (s->enable_order_hint) {
            s->enable_jnt_comp = f(&b, 1);
            s->enable_ref_frame_mvs = f(&b, 1);
        }
        if (f(&b, 1)) /* seq_choose_screen_content_tools */
            s->seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        else
            s->seq_force_screen_content_tools = f(&b, 1);
        if (s->seq_force_screen_content_tools > 0) {
            if (f(&b, 1)) /* seq_choose_integer_mv */
                s->seq_force_integer_mv = SELECT_INTEGER_MV;
            else
                s->seq_force_integer_mv = f(&b, 1);
        } else {
            s->seq_force_integer_mv = SELECT_INTEGER_MV;
        }
        if (s->enable_order_hint)
            s->order_hint_bits = f(&b, 3) + 1;
    }
    s->enable_superres = f(&b, 1);
    s->enable_cdef = f(&b, 1);
    s->enable_restoration = f(&b, 1);
    /* color_config() */
    const int high_bitdepth = f(&b, 1);
    if (s->seq_profile == 2 && high_bitdepth)
        s->bit_depth = f(&b, 1) ? 12 : 10;
    else
        s->bit_depth = high_bitdepth ? 10 : 8;
    s->mono_chrome = s->seq_profile == 1 ? 0 : (int)f(&b, 1);
    s->num_planes = s->mono_chrome ? 1 : 3;
    s->color_primaries = 2;
    s->transfer_characteristics = 2;
    s->matrix_coefficients = 2;
    if (f(&b, 1)) { /* color_description_present_flag */
        s->color_primaries = f(&b, 8);
        s->transfer_characteristics = f(&b, 8);
        s->matrix_coefficients = f(&b, 8);
    }
    if (s->mono_chrome) {
        s->color_range = f(&b, 1);
        s->subsampling_x = s->subsampling_y = 1;
        s->separate_uv_delta_q = 0;
    } else {
        if (s->color_primaries == 1 && s->transfer_characteristics == 13 && s->matrix_coefficients == 0) {
            s->color_range = 1;
            s->subsampling_x = s->subsampling_y = 0;
        } else {
            s->color_range = f(&b, 1);
            if (s->seq_profile == 0) {
                s->subsampling_x = s->subsampling_y = 1;
            } else if (s->seq_profile == 1) {
                s->subsampling_x = s->subsampling_y = 0;
            } else if (s->bit_depth == 12) {
                s->subsampling_x = f(&b, 1);
                s->subsampling_y = s->subsampling_x ? (int)f(&b, 1) : 0;
            } else {
                s->subsampling_x = 1;
                s->subsampling_y = 0;
            }
            if (s->subsampling_x && s->subsampling_y)
                s->chroma_sample_position = f(&b, 2);
        }
        s->separate_uv_delta_q = f(&b, 1);
    }
    s->film_grain_params_present = f(&b, 1);
    if (b.overrun) {
        seterr(err, errlen, "sequence header truncated");
        return -1;
    }
    return 0;
}

/* ---- frame header (5.9) ---- */

static int get_relative_dist(const Av1SeqHdr *s, int a, int b)
{
    if (!s->enable_order_hint)
        return 0;
    int diff = a - b;
    const int m = 1 << (s->order_hint_bits - 1);
    diff = (diff & (m - 1)) - (diff & m);
    return diff;
}

static void setup_past_independence(Av1Parser *p)
{
    static const int8_t def_ref_deltas[8] = { 1, 0, 0, 0, -1, 0, -1, -1 };
    memset(p->feature_enabled, 0, sizeof(p->feature_enabled));
    memset(p->feature_data, 0, sizeof(p->feature_data));
    for (int ref = 0; ref < 8; ref++)
        for (int i = 0; i < 6; i++)
            p->prev_gm_params[ref][i] = (i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0;
    memcpy(p->lf_ref_deltas, def_ref_deltas, 8);
    p->lf_mode_deltas[0] = p->lf_mode_deltas[1] = 0;
}

static void load_previous(Av1Parser *p, int slot)
{
    const Av1RefSlot *r = &p->ref[slot];
    memcpy(p->prev_gm_params, r->gm_params, sizeof(p->prev_gm_params));
    memcpy(p->lf_ref_deltas, r->loop_filter_ref_deltas, 8);
    memcpy(p->lf_mode_deltas, r->loop_filter_mode_deltas, 2);
    memcpy(p->feature_enabled, r->feature_enabled, sizeof(p->feature_enabled));
    memcpy(p->feature_data, r->feature_data, sizeof(p->feature_data));
}

static void superres_params(Av1Parser *p, BitReader *b, Av1FrameHdr *h)
{
    h->use_superres = p->seq.enable_superres ? (int)f(b, 1) : 0;
    h->superres_denom = h->use_superres ? (int)f(b, 3) + 9 : 8;
    h->upscaled_width = h->frame_width;
    h->frame_width = (h->upscaled_width * 8 + (h->superres_denom / 2)) / h->superres_denom;
}

static void frame_size(Av1Parser *p, BitReader *b, Av1FrameHdr *h)
{
    if (h->frame_size_override_flag) {
        h->frame_width = f(b, p->seq.frame_width_bits_minus_1 + 1) + 1;
        h->frame_height = f(b, p->seq.frame_height_bits_minus_1 + 1) + 1;
    } else {
        h->frame_width = p->seq.max_frame_width_minus_1 + 1;
        h->frame_height = p->seq.max_frame_height_minus_1 + 1;
    }
    superres_params(p, b, h);
}

static void render_size(BitReader *b, Av1FrameHdr *h)
{
    if (f(b, 1)) {
        h->render_width = f(b, 16) + 1;
        h->render_height = f(b, 16) + 1;
    } else {
        h->render_width = h->upscaled_width;
        h->render_height = h->frame_height;
    }
}

static int tile_log2(int blk, int target)
{
    int k = 0;
    while ((blk << k) < target)
        k++;
    return k;
}

static void tile_info(Av1Parser *p, BitReader *b, Av1FrameHdr *h)
{
    const int mi_cols = 2 * ((h->frame_width + 7) >> 3);
    const int mi_rows = 2 * ((h->frame_height + 7) >> 3);
    const int sb128 = p->seq.use_128x128_superblock;
    const int sb_cols = sb128 ? (mi_cols + 31) >> 5 : (mi_cols + 15) >> 4;
    const int sb_rows = sb128 ? (mi_rows + 31) >> 5 : (mi_rows + 15) >> 4;
    const int sb_shift = sb128 ? 5 : 4;
    const int sb_size = sb_shift + 2;
    const int max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    int max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    const int min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    const int max_log2_tile_cols = tile_log2(1, sb_cols < MAX_TILE_COLS ? sb_cols : MAX_TILE_COLS);
    const int max_log2_tile_rows = tile_log2(1, sb_rows < MAX_TILE_ROWS ? sb_rows : MAX_TILE_ROWS);
    const int t = tile_log2(max_tile_area_sb, sb_rows * sb_cols);
    const int min_log2_tiles = min_log2_tile_cols > t ? min_log2_tile_cols : t;
    if (f(b, 1)) { /* uniform_tile_spacing_flag */
        h->tile_cols_log2 = min_log2_tile_cols;
        while (h->tile_cols_log2 < max_log2_tile_cols && !b->overrun) {
            if (f(b, 1))
                h->tile_cols_log2++;
            else
                break;
        }
        const int tile_width_sb = (sb_cols + (1 << h->tile_cols_log2) - 1) >> h->tile_cols_log2;
        int i = 0;
        for (int start = 0; start < sb_cols; start += tile_width_sb)
            i++;
        h->tile_cols = i;
        const int min_log2_tile_rows = min_log2_tiles - h->tile_cols_log2 > 0 ? min_log2_tiles - h->tile_cols_log2 : 0;
        h->tile_rows_log2 = min_log2_tile_rows;
        while (h->tile_rows_log2 < max_log2_tile_rows && !b->overrun) {
            if (f(b, 1))
                h->tile_rows_log2++;
            else
                break;
        }
        const int tile_height_sb = (sb_rows + (1 << h->tile_rows_log2) - 1) >> h->tile_rows_log2;
        i = 0;
        for (int start = 0; start < sb_rows; start += tile_height_sb)
            i++;
        h->tile_rows = i;
    } else {
        int widest = 0, start = 0, i;
        for (i = 0; start < sb_cols && !b->overrun; i++) {
            const int max_width = sb_cols - start < max_tile_width_sb ? sb_cols - start : max_tile_width_sb;
            const int size = (int)ns(b, max_width) + 1;
            if (size > widest)
                widest = size;
            start += size;
        }
        h->tile_cols = i;
        h->tile_cols_log2 = tile_log2(1, h->tile_cols);
        if (min_log2_tiles > 0)
            max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
        else
            max_tile_area_sb = sb_rows * sb_cols;
        int max_tile_height_sb = widest ? max_tile_area_sb / widest : 1;
        if (max_tile_height_sb < 1)
            max_tile_height_sb = 1;
        start = 0;
        for (i = 0; start < sb_rows && !b->overrun; i++) {
            const int max_height = sb_rows - start < max_tile_height_sb ? sb_rows - start : max_tile_height_sb;
            start += (int)ns(b, max_height) + 1;
        }
        h->tile_rows = i;
        h->tile_rows_log2 = tile_log2(1, h->tile_rows);
    }
    if (h->tile_cols_log2 > 0 || h->tile_rows_log2 > 0) {
        h->context_update_tile_id = f(b, h->tile_rows_log2 + h->tile_cols_log2);
        h->tile_size_bytes = f(b, 2) + 1;
    } else {
        h->context_update_tile_id = 0;
        h->tile_size_bytes = 4;
    }
}

static int read_delta_q(BitReader *b)
{
    return f(b, 1) ? su(b, 7) : 0;
}

static void set_frame_refs(Av1Parser *p, Av1FrameHdr *h, int last_frame_idx, int gold_frame_idx)
{
    const Av1SeqHdr *s = &p->seq;
    int used[8] = { 0 }, shifted[8];
    for (int i = 0; i < REFS_PER_FRAME; i++)
        h->ref_frame_idx[i] = -1;
    h->ref_frame_idx[0] = last_frame_idx;
    h->ref_frame_idx[3] = gold_frame_idx; /* GOLDEN_FRAME - LAST_FRAME */
    used[last_frame_idx] = used[gold_frame_idx] = 1;
    const int cur = 1 << (s->order_hint_bits - 1);
    for (int i = 0; i < NUM_REF_FRAMES; i++)
        shifted[i] = cur + get_relative_dist(s, p->ref[i].order_hint, h->order_hint);
    int ref = -1, hint, best = 0;
    /* ALTREF: latest backward */
    for (int i = 0; i < 8; i++) {
        hint = shifted[i];
        if (!used[i] && hint >= cur && (ref < 0 || hint >= best)) {
            ref = i;
            best = hint;
        }
    }
    if (ref >= 0) {
        h->ref_frame_idx[6] = ref;
        used[ref] = 1;
    }
    /* BWDREF, then ALTREF2: earliest backward */
    for (int k = 0; k < 2; k++) {
        ref = -1;
        for (int i = 0; i < 8; i++) {
            hint = shifted[i];
            if (!used[i] && hint >= cur && (ref < 0 || hint < best)) {
                ref = i;
                best = hint;
            }
        }
        if (ref >= 0) {
            h->ref_frame_idx[k == 0 ? 4 : 5] = ref;
            used[ref] = 1;
        }
    }
    /* LAST2, LAST3, BWDREF, ALTREF2, ALTREF: latest forward */
    static const int ref_list[5] = { 1, 2, 4, 5, 6 };
    for (int k = 0; k < 5; k++) {
        const int rf = ref_list[k];
        if (h->ref_frame_idx[rf] >= 0)
            continue;
        ref = -1;
        for (int i = 0; i < 8; i++) {
            hint = shifted[i];
            if (!used[i] && hint < cur && (ref < 0 || hint >= best)) {
                ref = i;
                best = hint;
            }
        }
        if (ref >= 0) {
            h->ref_frame_idx[rf] = ref;
            used[ref] = 1;
        }
    }
    ref = -1;
    for (int i = 0; i < 8; i++) {
        hint = shifted[i];
        if (ref < 0 || hint < best) {
            ref = i;
            best = hint;
        }
    }
    for (int i = 0; i < REFS_PER_FRAME; i++)
        if (h->ref_frame_idx[i] < 0)
            h->ref_frame_idx[i] = ref;
}

static int inverse_recenter(int r, int v)
{
    if (v > 2 * r)
        return v;
    else if (v & 1)
        return r - ((v + 1) >> 1);
    else
        return r + (v >> 1);
}

static int decode_subexp(BitReader *b, int num_syms)
{
    int i = 0, mk = 0;
    const int k = 3;
    while (!b->overrun) {
        const int b2 = i ? k + i - 1 : k;
        const int a = 1 << b2;
        if (num_syms <= mk + 3 * a) {
            return (int)ns(b, num_syms - mk) + mk;
        } else if (f(b, 1)) {
            i++;
            mk += a;
        } else {
            return (int)f(b, b2) + mk;
        }
    }
    return 0;
}

static int decode_signed_subexp_with_ref(BitReader *b, int low, int high, int r)
{
    const int mx = high - low;
    r -= low;
    const int v = decode_subexp(b, mx);
    const int x = (r << 1) <= mx ? inverse_recenter(r, v) : mx - 1 - inverse_recenter(mx - 1 - r, v);
    return x + low;
}

static void read_global_param(Av1Parser *p, BitReader *b, Av1FrameHdr *h, int type, int ref, int idx)
{
    int abs_bits = GM_ABS_ALPHA_BITS, prec_bits = GM_ALPHA_PREC_BITS;
    if (idx < 2) {
        if (type == GM_TRANSLATION) {
            abs_bits = GM_ABS_TRANS_ONLY_BITS - !h->allow_high_precision_mv;
            prec_bits = GM_TRANS_ONLY_PREC_BITS - !h->allow_high_precision_mv;
        } else {
            abs_bits = GM_ABS_TRANS_BITS;
            prec_bits = GM_TRANS_PREC_BITS;
        }
    }
    const int prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    const int round = (idx % 3) == 2 ? (1 << WARPEDMODEL_PREC_BITS) : 0;
    const int sub = (idx % 3) == 2 ? (1 << prec_bits) : 0;
    const int mx = 1 << abs_bits;
    const int r = (p->prev_gm_params[ref][idx] >> prec_diff) - sub;
    h->gm_params[ref][idx] = (int32_t)((unsigned)decode_signed_subexp_with_ref(b, -mx, mx + 1, r) << prec_diff) + round;
}

static void reset_grain(Av1FilmGrain *g)
{
    memset(g, 0, sizeof(*g));
}

static int film_grain_params(Av1Parser *p, BitReader *b, Av1FrameHdr *h)
{
    const Av1SeqHdr *s = &p->seq;
    Av1FilmGrain *g = &h->fg;
    h->fg_bit_offset = -1;
    h->seed_bit_offset = -1;
    h->fg_parsed = 0;
    if (!s->film_grain_params_present || (!h->show_frame && !h->showable_frame)) {
        reset_grain(g);
        h->fg_bit_end = b->pos;
        return 0;
    }
    h->fg_parsed = 1;
    h->fg_bit_offset = b->pos;
    g->apply_grain = f(b, 1);
    if (!g->apply_grain) {
        reset_grain(g);
        h->fg_bit_end = b->pos;
        return 0;
    }
    h->seed_bit_offset = b->pos;
    const unsigned seed = f(b, 16);
    g->grain_seed = (uint16_t)seed;
    g->update_grain = h->frame_type == AV1_INTER_FRAME ? f(b, 1) : 1;
    h->update_grain_coded = g->update_grain;
    if (!g->update_grain) {
        const int idx = f(b, 3);
        int ok = 0;
        for (int i = 0; i < REFS_PER_FRAME; i++)
            ok |= h->ref_frame_idx[i] == idx;
        if (!ok && p->strict) {
            seterr(p->err, sizeof(p->err), "film_grain_params_ref_idx %d is not a reference of this frame", idx);
            return -1;
        }
        *g = p->ref[idx].fg;
        g->grain_seed = (uint16_t)seed;
        g->film_grain_params_ref_idx = (uint8_t)idx;
        h->fg_bit_end = b->pos;
        return 0;
    }
    g->num_y_points = f(b, 4);
    if (g->num_y_points > 14) {
        seterr(p->err, sizeof(p->err), "num_y_points %d > 14", g->num_y_points);
        return -1;
    }
    for (int i = 0; i < g->num_y_points; i++) {
        g->point_y_value[i] = f(b, 8);
        g->point_y_scaling[i] = f(b, 8);
    }
    g->chroma_scaling_from_luma = s->mono_chrome ? 0 : f(b, 1);
    if (s->mono_chrome || g->chroma_scaling_from_luma ||
        (s->subsampling_x == 1 && s->subsampling_y == 1 && g->num_y_points == 0)) {
        g->num_cb_points = g->num_cr_points = 0;
    } else {
        g->num_cb_points = f(b, 4);
        if (g->num_cb_points > 10) {
            seterr(p->err, sizeof(p->err), "num_cb_points %d > 10", g->num_cb_points);
            return -1;
        }
        for (int i = 0; i < g->num_cb_points; i++) {
            g->point_cb_value[i] = f(b, 8);
            g->point_cb_scaling[i] = f(b, 8);
        }
        g->num_cr_points = f(b, 4);
        if (g->num_cr_points > 10) {
            seterr(p->err, sizeof(p->err), "num_cr_points %d > 10", g->num_cr_points);
            return -1;
        }
        for (int i = 0; i < g->num_cr_points; i++) {
            g->point_cr_value[i] = f(b, 8);
            g->point_cr_scaling[i] = f(b, 8);
        }
    }
    g->grain_scaling_minus_8 = f(b, 2);
    g->ar_coeff_lag = f(b, 2);
    const int num_pos_luma = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
    int num_pos_chroma = num_pos_luma;
    if (g->num_y_points) {
        num_pos_chroma = num_pos_luma + 1;
        for (int i = 0; i < num_pos_luma; i++)
            g->ar_coeffs_y_plus_128[i] = f(b, 8);
    }
    if (g->chroma_scaling_from_luma || g->num_cb_points)
        for (int i = 0; i < num_pos_chroma; i++)
            g->ar_coeffs_cb_plus_128[i] = f(b, 8);
    if (g->chroma_scaling_from_luma || g->num_cr_points)
        for (int i = 0; i < num_pos_chroma; i++)
            g->ar_coeffs_cr_plus_128[i] = f(b, 8);
    g->ar_coeff_shift_minus_6 = f(b, 2);
    g->grain_scale_shift = f(b, 2);
    if (g->num_cb_points) {
        g->cb_mult = f(b, 8);
        g->cb_luma_mult = f(b, 8);
        g->cb_offset = f(b, 9);
    }
    if (g->num_cr_points) {
        g->cr_mult = f(b, 8);
        g->cr_luma_mult = f(b, 8);
        g->cr_offset = f(b, 9);
    }
    g->overlap_flag = f(b, 1);
    g->clip_to_restricted_range = f(b, 1);
    h->fg_bit_end = b->pos;
    return 0;
}

/* reference frame update process (7.20) */
static void reference_update(Av1Parser *p, const Av1FrameHdr *h)
{
    for (int i = 0; i < NUM_REF_FRAMES; i++) {
        if (!((h->refresh_frame_flags >> i) & 1))
            continue;
        Av1RefSlot *r = &p->ref[i];
        r->valid = 1;
        r->frame_id = h->current_frame_id;
        r->frame_type = h->frame_type;
        r->order_hint = h->order_hint;
        r->showable = h->showable_frame;
        r->upscaled_width = h->upscaled_width;
        r->frame_width = h->frame_width;
        r->frame_height = h->frame_height;
        r->render_width = h->render_width;
        r->render_height = h->render_height;
        memcpy(r->saved_order_hints, p->order_hints, sizeof(r->saved_order_hints));
        memcpy(r->gm_params, h->gm_params, sizeof(r->gm_params));
        memcpy(r->loop_filter_ref_deltas, p->lf_ref_deltas, 8);
        memcpy(r->loop_filter_mode_deltas, p->lf_mode_deltas, 2);
        memcpy(r->feature_enabled, p->feature_enabled, sizeof(r->feature_enabled));
        memcpy(r->feature_data, p->feature_data, sizeof(r->feature_data));
        r->fg = h->fg;
        r->bit_depth = p->seq.bit_depth;
        r->subsampling_x = p->seq.subsampling_x;
        r->subsampling_y = p->seq.subsampling_y;
    }
}

static int uncompressed_header(Av1Parser *p, BitReader *b, Av1FrameHdr *h)
{
    const Av1SeqHdr *s = &p->seq;
    int id_len = 0;
    if (s->frame_id_numbers_present_flag)
        id_len = s->additional_frame_id_length_minus_1 + s->delta_frame_id_length_minus_2 + 3;
    const int all_frames = (1 << NUM_REF_FRAMES) - 1;
    int frame_is_intra;
    h->fg_bit_offset = h->seed_bit_offset = -1;
    if (s->reduced_still_picture_header) {
        h->show_existing_frame = 0;
        h->frame_type = AV1_KEY_FRAME;
        frame_is_intra = 1;
        h->show_frame = 1;
        h->showable_frame = 0;
        h->error_resilient_mode = 1;
    } else {
        h->show_existing_frame = f(b, 1);
        if (h->show_existing_frame) {
            h->frame_to_show_map_idx = f(b, 3);
            if (s->decoder_model_info_present_flag && !s->equal_picture_interval)
                f(b, s->frame_presentation_time_length_minus_1 + 1);
            h->refresh_frame_flags = 0;
            if (s->frame_id_numbers_present_flag)
                f(b, id_len); /* display_frame_id */
            const Av1RefSlot *r = &p->ref[h->frame_to_show_map_idx];
            if (!r->valid && p->strict) {
                seterr(p->err, sizeof(p->err), "show_existing_frame of an empty slot %d", h->frame_to_show_map_idx);
                return -1;
            }
            h->frame_type = r->frame_type;
            if (h->frame_type == AV1_KEY_FRAME)
                h->refresh_frame_flags = all_frames;
            h->show_frame = 1;
            h->showable_frame = r->showable;
            h->order_hint = r->order_hint;
            h->upscaled_width = r->upscaled_width;
            h->frame_width = r->frame_width;
            h->frame_height = r->frame_height;
            h->render_width = r->render_width;
            h->render_height = r->render_height;
            h->current_frame_id = r->frame_id;
            if (s->film_grain_params_present)
                h->fg = r->fg;
            h->header_bits = b->pos;
            return 0;
        }
        h->frame_type = f(b, 2);
        frame_is_intra = h->frame_type == AV1_INTRA_ONLY_FRAME || h->frame_type == AV1_KEY_FRAME;
        h->show_frame = f(b, 1);
        if (h->show_frame && s->decoder_model_info_present_flag && !s->equal_picture_interval)
            f(b, s->frame_presentation_time_length_minus_1 + 1);
        if (h->show_frame)
            h->showable_frame = h->frame_type != AV1_KEY_FRAME;
        else
            h->showable_frame = f(b, 1);
        if (h->frame_type == AV1_SWITCH_FRAME || (h->frame_type == AV1_KEY_FRAME && h->show_frame))
            h->error_resilient_mode = 1;
        else
            h->error_resilient_mode = f(b, 1);
    }
    if (h->frame_type == AV1_KEY_FRAME && h->show_frame) {
        for (int i = 0; i < NUM_REF_FRAMES; i++) {
            p->ref[i].valid = 0;
            p->ref[i].order_hint = 0;
        }
        for (int i = 0; i < 8; i++)
            p->order_hints[i] = 0;
    }
    h->disable_cdf_update = f(b, 1);
    if (s->seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS)
        h->allow_screen_content_tools = f(b, 1);
    else
        h->allow_screen_content_tools = s->seq_force_screen_content_tools;
    if (h->allow_screen_content_tools) {
        if (s->seq_force_integer_mv == SELECT_INTEGER_MV)
            h->force_integer_mv = f(b, 1);
        else
            h->force_integer_mv = s->seq_force_integer_mv;
    } else {
        h->force_integer_mv = 0;
    }
    if (frame_is_intra)
        h->force_integer_mv = 1;
    if (s->frame_id_numbers_present_flag) {
        h->current_frame_id = f(b, id_len);
        /* mark_ref_frames() */
        const int diff_len = s->delta_frame_id_length_minus_2 + 2;
        for (int i = 0; i < NUM_REF_FRAMES; i++) {
            const int cid = h->current_frame_id, rid = p->ref[i].frame_id;
            if (h->frame_type == AV1_KEY_FRAME && h->show_frame)
                break;
            if (cid > (1 << diff_len)) {
                if (rid > cid || rid < cid - (1 << diff_len))
                    p->ref[i].valid = 0;
            } else if (rid > cid && rid < (1 << id_len) + cid - (1 << diff_len)) {
                p->ref[i].valid = 0;
            }
        }
    } else {
        h->current_frame_id = 0;
    }
    if (h->frame_type == AV1_SWITCH_FRAME)
        h->frame_size_override_flag = 1;
    else if (s->reduced_still_picture_header)
        h->frame_size_override_flag = 0;
    else
        h->frame_size_override_flag = f(b, 1);
    h->order_hint = f(b, s->order_hint_bits);
    if (frame_is_intra || h->error_resilient_mode)
        h->primary_ref_frame = PRIMARY_REF_NONE;
    else
        h->primary_ref_frame = f(b, 3);
    if (s->decoder_model_info_present_flag) {
        if (f(b, 1)) { /* buffer_removal_time_present_flag */
            for (int op = 0; op <= s->operating_points_cnt_minus_1; op++) {
                if (!s->decoder_model_present_for_this_op[op])
                    continue;
                const int idc = s->operating_point_idc[op];
                const int in_t = (idc >> h->temporal_id) & 1;
                const int in_s = (idc >> (h->spatial_id + 8)) & 1;
                if (idc == 0 || (in_t && in_s))
                    f(b, s->buffer_removal_time_length_minus_1 + 1);
            }
        }
    }
    h->allow_high_precision_mv = 0;
    h->use_ref_frame_mvs = 0;
    h->allow_intrabc = 0;
    if (h->frame_type == AV1_SWITCH_FRAME || (h->frame_type == AV1_KEY_FRAME && h->show_frame))
        h->refresh_frame_flags = all_frames;
    else
        h->refresh_frame_flags = f(b, 8);
    if (!frame_is_intra || h->refresh_frame_flags != all_frames) {
        if (h->error_resilient_mode && s->enable_order_hint) {
            for (int i = 0; i < NUM_REF_FRAMES; i++) {
                const int roh = f(b, s->order_hint_bits);
                if (roh != p->ref[i].order_hint || !p->ref[i].valid) {
                    /* the reference becomes an (invalid) frame with this order hint */
                    p->ref[i].order_hint = roh;
                    p->ref[i].valid = 0;
                }
            }
        }
    }
    for (int i = 0; i < REFS_PER_FRAME; i++)
        h->ref_frame_idx[i] = 0;
    if (frame_is_intra) {
        frame_size(p, b, h);
        render_size(b, h);
        if (h->allow_screen_content_tools && h->upscaled_width == h->frame_width)
            h->allow_intrabc = f(b, 1);
    } else {
        int short_signaling = 0;
        if (s->enable_order_hint) {
            short_signaling = f(b, 1);
            if (short_signaling) {
                const int last = f(b, 3), gold = f(b, 3);
                set_frame_refs(p, h, last, gold);
            }
        }
        for (int i = 0; i < REFS_PER_FRAME; i++) {
            if (!short_signaling)
                h->ref_frame_idx[i] = f(b, 3);
            if (s->frame_id_numbers_present_flag)
                f(b, s->delta_frame_id_length_minus_2 + 2);
        }
        if (h->frame_size_override_flag && !h->error_resilient_mode) {
            int found = 0;
            for (int i = 0; i < REFS_PER_FRAME && !b->overrun; i++) {
                if (f(b, 1)) {
                    const Av1RefSlot *r = &p->ref[h->ref_frame_idx[i]];
                    h->upscaled_width = r->upscaled_width;
                    h->frame_width = h->upscaled_width;
                    h->frame_height = r->frame_height;
                    h->render_width = r->render_width;
                    h->render_height = r->render_height;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                frame_size(p, b, h);
                render_size(b, h);
            } else {
                superres_params(p, b, h);
            }
        } else {
            frame_size(p, b, h);
            render_size(b, h);
        }
        h->allow_high_precision_mv = h->force_integer_mv ? 0 : (int)f(b, 1);
        h->is_filter_switchable = f(b, 1);
        h->interpolation_filter = h->is_filter_switchable ? 4 : (int)f(b, 2);
        h->is_motion_mode_switchable = f(b, 1);
        if (h->error_resilient_mode || !s->enable_ref_frame_mvs)
            h->use_ref_frame_mvs = 0;
        else
            h->use_ref_frame_mvs = f(b, 1);
        for (int i = 0; i < REFS_PER_FRAME; i++)
            p->order_hints[1 + i] = p->ref[h->ref_frame_idx[i]].order_hint;
    }
    if (h->frame_width <= 0 || h->frame_height <= 0) {
        seterr(p->err, sizeof(p->err), "frame size %dx%d", h->frame_width, h->frame_height);
        return -1;
    }
    p->order_hints[0] = h->order_hint;
    if (s->reduced_still_picture_header || h->disable_cdf_update)
        h->disable_frame_end_update_cdf = 1;
    else
        h->disable_frame_end_update_cdf = f(b, 1);
    if (h->primary_ref_frame == PRIMARY_REF_NONE)
        setup_past_independence(p);
    else
        load_previous(p, h->ref_frame_idx[h->primary_ref_frame]);
    tile_info(p, b, h);
    /* quantization_params() */
    h->base_q_idx = f(b, 8);
    h->delta_q_y_dc = read_delta_q(b);
    h->delta_q_u_dc = h->delta_q_u_ac = h->delta_q_v_dc = h->delta_q_v_ac = 0;
    if (s->num_planes > 1) {
        const int diff_uv_delta = s->separate_uv_delta_q ? (int)f(b, 1) : 0;
        h->delta_q_u_dc = read_delta_q(b);
        h->delta_q_u_ac = read_delta_q(b);
        if (diff_uv_delta) {
            h->delta_q_v_dc = read_delta_q(b);
            h->delta_q_v_ac = read_delta_q(b);
        } else {
            h->delta_q_v_dc = h->delta_q_u_dc;
            h->delta_q_v_ac = h->delta_q_u_ac;
        }
    }
    h->using_qmatrix = f(b, 1);
    if (h->using_qmatrix) {
        f(b, 4);
        f(b, 4);
        if (s->separate_uv_delta_q)
            f(b, 4);
    }
    /* segmentation_params() */
    static const int seg_bits[8] = { 8, 6, 6, 6, 6, 3, 0, 0 };
    static const int seg_signed[8] = { 1, 1, 1, 1, 1, 0, 0, 0 };
    static const int seg_max[8] = { 255, 63, 63, 63, 63, 7, 0, 0 };
    h->segmentation_enabled = f(b, 1);
    if (h->segmentation_enabled) {
        int update_data = 1;
        if (h->primary_ref_frame != PRIMARY_REF_NONE) {
            if (f(b, 1)) /* segmentation_update_map */
                f(b, 1); /* segmentation_temporal_update */
            update_data = f(b, 1);
        }
        if (update_data) {
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) {
                    int v = 0;
                    const int en = f(b, 1);
                    p->feature_enabled[i][j] = (uint8_t)en;
                    if (en) {
                        const int lim = seg_max[j];
                        if (seg_signed[j]) {
                            v = su(b, 1 + seg_bits[j]);
                            v = v < -lim ? -lim : v > lim ? lim : v;
                        } else {
                            v = f(b, seg_bits[j]);
                            v = v > lim ? lim : v;
                        }
                    }
                    p->feature_data[i][j] = (int16_t)v;
                }
        }
    } else {
        memset(p->feature_enabled, 0, sizeof(p->feature_enabled));
        memset(p->feature_data, 0, sizeof(p->feature_data));
    }
    /* delta_q_params(), delta_lf_params() */
    h->delta_q_present = 0;
    if (h->base_q_idx > 0)
        h->delta_q_present = f(b, 1);
    if (h->delta_q_present)
        f(b, 2); /* delta_q_res */
    h->delta_lf_present = 0;
    if (h->delta_q_present) {
        if (!h->allow_intrabc)
            h->delta_lf_present = f(b, 1);
        if (h->delta_lf_present) {
            f(b, 2); /* delta_lf_res */
            f(b, 1); /* delta_lf_multi */
        }
    }
    h->coded_lossless = 1;
    for (int seg = 0; seg < 8; seg++) {
        int q = h->base_q_idx;
        if (h->segmentation_enabled && p->feature_enabled[seg][0]) {
            q += p->feature_data[seg][0];
            q = q < 0 ? 0 : q > 255 ? 255 : q;
        }
        const int lossless = q == 0 && !h->delta_q_y_dc && !h->delta_q_u_ac && !h->delta_q_u_dc &&
                             !h->delta_q_v_ac && !h->delta_q_v_dc;
        if (!lossless)
            h->coded_lossless = 0;
    }
    h->all_lossless = h->coded_lossless && h->frame_width == h->upscaled_width;
    /* loop_filter_params() */
    if (h->coded_lossless || h->allow_intrabc) {
        static const int8_t def_ref_deltas[8] = { 1, 0, 0, 0, -1, 0, -1, -1 };
        h->loop_filter_level[0] = h->loop_filter_level[1] = 0;
        memcpy(p->lf_ref_deltas, def_ref_deltas, 8);
        p->lf_mode_deltas[0] = p->lf_mode_deltas[1] = 0;
    } else {
        h->loop_filter_level[0] = f(b, 6);
        h->loop_filter_level[1] = f(b, 6);
        if (s->num_planes > 1 && (h->loop_filter_level[0] || h->loop_filter_level[1])) {
            h->loop_filter_level[2] = f(b, 6);
            h->loop_filter_level[3] = f(b, 6);
        }
        f(b, 3); /* loop_filter_sharpness */
        if (f(b, 1)) { /* loop_filter_delta_enabled */
            if (f(b, 1)) { /* loop_filter_delta_update */
                for (int i = 0; i < 8; i++)
                    if (f(b, 1))
                        p->lf_ref_deltas[i] = (int8_t)su(b, 7);
                for (int i = 0; i < 2; i++)
                    if (f(b, 1))
                        p->lf_mode_deltas[i] = (int8_t)su(b, 7);
            }
        }
    }
    /* cdef_params() */
    h->cdef_bits = 0;
    if (!(h->coded_lossless || h->allow_intrabc || !s->enable_cdef)) {
        f(b, 2); /* cdef_damping_minus_3 */
        h->cdef_bits = f(b, 2);
        for (int i = 0; i < (1 << h->cdef_bits); i++) {
            f(b, 4);
            f(b, 2);
            if (s->num_planes > 1) {
                f(b, 4);
                f(b, 2);
            }
        }
    }
    /* lr_params() */
    if (!(h->all_lossless || h->allow_intrabc || !s->enable_restoration)) {
        int uses_lr = 0, uses_chroma_lr = 0;
        for (int i = 0; i < s->num_planes; i++) {
            const int lr_type = f(b, 2);
            if (lr_type) {
                uses_lr = 1;
                if (i > 0)
                    uses_chroma_lr = 1;
            }
        }
        if (uses_lr) {
            if (s->use_128x128_superblock) {
                f(b, 1);
            } else if (f(b, 1)) {
                f(b, 1);
            }
            if (s->subsampling_x && s->subsampling_y && uses_chroma_lr)
                f(b, 1);
        }
    }
    /* read_tx_mode() */
    h->tx_mode_select = h->coded_lossless ? 0 : (int)f(b, 1);
    /* frame_reference_mode() */
    h->reference_select = frame_is_intra ? 0 : (int)f(b, 1);
    /* skip_mode_params() */
    int skip_mode_allowed = 0;
    if (!(frame_is_intra || !h->reference_select || !s->enable_order_hint)) {
        int fwd = -1, bwd = -1, fwd_hint = 0, bwd_hint = 0;
        for (int i = 0; i < REFS_PER_FRAME; i++) {
            const int ref_hint = p->ref[h->ref_frame_idx[i]].order_hint;
            if (get_relative_dist(s, ref_hint, h->order_hint) < 0) {
                if (fwd < 0 || get_relative_dist(s, ref_hint, fwd_hint) > 0) {
                    fwd = i;
                    fwd_hint = ref_hint;
                }
            } else if (get_relative_dist(s, ref_hint, h->order_hint) > 0) {
                if (bwd < 0 || get_relative_dist(s, ref_hint, bwd_hint) < 0) {
                    bwd = i;
                    bwd_hint = ref_hint;
                }
            }
        }
        if (fwd < 0) {
            skip_mode_allowed = 0;
        } else if (bwd >= 0) {
            skip_mode_allowed = 1;
        } else {
            int fwd2 = -1, fwd2_hint = 0;
            for (int i = 0; i < REFS_PER_FRAME; i++) {
                const int ref_hint = p->ref[h->ref_frame_idx[i]].order_hint;
                if (get_relative_dist(s, ref_hint, fwd_hint) < 0) {
                    if (fwd2 < 0 || get_relative_dist(s, ref_hint, fwd2_hint) > 0) {
                        fwd2 = i;
                        fwd2_hint = ref_hint;
                    }
                }
            }
            skip_mode_allowed = fwd2 >= 0;
        }
    }
    h->skip_mode_present = skip_mode_allowed ? (int)f(b, 1) : 0;
    if (frame_is_intra || h->error_resilient_mode || !s->enable_warped_motion)
        h->allow_warped_motion = 0;
    else
        h->allow_warped_motion = f(b, 1);
    h->reduced_tx_set = f(b, 1);
    /* global_motion_params() */
    for (int ref = 0; ref < 8; ref++) {
        h->gm_type[ref] = GM_IDENTITY;
        for (int i = 0; i < 6; i++)
            h->gm_params[ref][i] = (i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0;
    }
    if (!frame_is_intra) {
        for (int ref = 1; ref <= 7; ref++) {
            int type = GM_IDENTITY;
            if (f(b, 1)) { /* is_global */
                if (f(b, 1)) /* is_rot_zoom */
                    type = GM_ROTZOOM;
                else
                    type = f(b, 1) ? GM_TRANSLATION : GM_AFFINE;
            }
            h->gm_type[ref] = type;
            if (type >= GM_ROTZOOM) {
                read_global_param(p, b, h, type, ref, 2);
                read_global_param(p, b, h, type, ref, 3);
                if (type == GM_AFFINE) {
                    read_global_param(p, b, h, type, ref, 4);
                    read_global_param(p, b, h, type, ref, 5);
                } else {
                    h->gm_params[ref][4] = -h->gm_params[ref][3];
                    h->gm_params[ref][5] = h->gm_params[ref][2];
                }
            }
            if (type >= GM_TRANSLATION) {
                read_global_param(p, b, h, type, ref, 0);
                read_global_param(p, b, h, type, ref, 1);
            }
        }
    }
    if (film_grain_params(p, b, h) < 0)
        return -1;
    h->header_bits = b->pos;
    return 0;
}

void av1_parser_init(Av1Parser *p)
{
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < NUM_REF_FRAMES; i++)
        for (int ref = 0; ref < 8; ref++)
            for (int k = 0; k < 6; k++)
                p->ref[i].gm_params[ref][k] = (k % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0;
    setup_past_independence(p);
}

int av1_obu_header(Av1Parser *p, const uint8_t *buf, size_t len, size_t size_hint, Av1Obu *obu)
{
    memset(obu, 0, sizeof(*obu));
    if (len < 1) {
        seterr(p->err, sizeof(p->err), "empty OBU");
        return -1;
    }
    const uint8_t h0 = buf[0];
    if (h0 & 0x80) {
        seterr(p->err, sizeof(p->err), "obu_forbidden_bit set (not an AV1 OBU stream?)");
        return -1;
    }
    obu->type = (h0 >> 3) & 15;
    obu->has_extension = (h0 >> 2) & 1;
    obu->has_size_field = (h0 >> 1) & 1;
    size_t pos = 1;
    if (obu->has_extension) {
        if (len < 2) {
            seterr(p->err, sizeof(p->err), "truncated OBU extension");
            return -1;
        }
        obu->temporal_id = buf[1] >> 5;
        obu->spatial_id = (buf[1] >> 3) & 3;
        pos = 2;
    }
    if (obu->has_size_field) {
        uint64_t sz;
        const int n = av1_leb128(buf + pos, len - pos, &sz);
        if (!n) {
            seterr(p->err, sizeof(p->err), "bad obu_size");
            return -1;
        }
        pos += n;
        obu->payload_len = (size_t)sz;
        if (size_hint && pos + sz > size_hint) {
            seterr(p->err, sizeof(p->err), "obu_size %llu exceeds the unit (%zu bytes left)",
                   (unsigned long long)sz, size_hint - pos);
            return -1;
        }
    } else {
        if (!size_hint || size_hint < pos) {
            seterr(p->err, sizeof(p->err), "OBU without obu_size and no unit size");
            return -1;
        }
        obu->payload_len = size_hint - pos;
    }
    obu->header_len = pos;
    obu->total_len = pos + obu->payload_len;
    return 0;
}

static int frame_header_obu(Av1Parser *p, const uint8_t *pl, size_t avail, size_t full, Av1Obu *obu, BitReader *b)
{
    Av1FrameHdr *h = &obu->frame_hdr;
    if (p->seen_frame_header) {
        /* frame_header_copy(): must repeat the current header bit for bit */
        *h = p->cur;
        h->obu_type = obu->type;
        h->is_copy = 1;
        const int64_t nb = p->cur.header_bits;
        if ((int64_t)avail * 8 < nb) {
            return (int64_t)full * 8 < nb ? -1 : -2;
        }
        const size_t nbytes = (size_t)(nb >> 3);
        const size_t cmp = nbytes < p->cur_hdr_copy_len ? nbytes : p->cur_hdr_copy_len;
        int same = !memcmp(pl, p->cur_hdr_copy, cmp);
        if (same && (nb & 7) && nbytes < p->cur_hdr_copy_len) {
            const uint8_t mask = (uint8_t)(0xff00 >> (nb & 7));
            same = (pl[nbytes] & mask) == (p->cur_hdr_copy[nbytes] & mask);
        }
        if (!same) {
            seterr(p->err, sizeof(p->err), "frame header copy differs from the frame header");
            if (p->strict)
                return -1;
        }
        b->pos = nb;
        obu->has_frame_hdr = 1;
        return 0;
    }
    memset(h, 0, sizeof(*h));
    h->obu_type = obu->type;
    h->temporal_id = obu->temporal_id;
    h->spatial_id = obu->spatial_id;
    if (uncompressed_header(p, b, h) < 0)
        return -1;
    if (b->overrun) {
        seterr(p->err, sizeof(p->err), "frame header truncated");
        return b->nbits < b->full_bits ? -2 : -1;
    }
    obu->has_frame_hdr = 1;
    p->cur = *h;
    p->cur_hdr_copy_len = avail < sizeof(p->cur_hdr_copy) ? avail : sizeof(p->cur_hdr_copy);
    memcpy(p->cur_hdr_copy, pl, p->cur_hdr_copy_len);
    if (h->show_existing_frame) {
        /* decode_frame_wrapup(): a shown key frame is loaded and refreshes every slot */
        if (h->frame_type == AV1_KEY_FRAME) {
            const Av1RefSlot r = p->ref[h->frame_to_show_map_idx];
            for (int i = 0; i < NUM_REF_FRAMES; i++)
                p->ref[i] = r;
        }
        p->seen_frame_header = 0;
        obu->frame_end = 1;
    } else {
        p->seen_frame_header = 1;
        p->tiles_seen = 0;
    }
    return 0;
}

static int tile_group_obu(Av1Parser *p, BitReader *b, Av1Obu *obu)
{
    if (!p->seen_frame_header) {
        seterr(p->err, sizeof(p->err), "tile group without a frame header");
        return p->strict ? -1 : 0;
    }
    const Av1FrameHdr *h = &p->cur;
    const int num_tiles = h->tile_cols * h->tile_rows;
    int tg_start = 0, tg_end = num_tiles - 1;
    if (num_tiles > 1 && f(b, 1)) {
        const int bits = h->tile_cols_log2 + h->tile_rows_log2;
        tg_start = f(b, bits);
        tg_end = f(b, bits);
    }
    if (b->overrun)
        return b->nbits < b->full_bits ? -2 : -1;
    (void)tg_start;
    if (tg_end == num_tiles - 1) {
        reference_update(p, &p->cur);
        p->seen_frame_header = 0;
        obu->frame_end = 1;
    }
    return 0;
}

int av1_parse_obu(Av1Parser *p, const uint8_t *buf, size_t avail, size_t size_hint, Av1Obu *obu)
{
    if (av1_obu_header(p, buf, avail, size_hint, obu) < 0)
        return -1;
    const uint8_t *pl = buf + obu->header_len;
    const size_t pavail = avail > obu->header_len ? avail - obu->header_len : 0;
    BitReader b;
    br_init(&b, pl, pavail, obu->payload_len);

    if (obu->type != AV1_OBU_SEQUENCE_HEADER && obu->type != AV1_OBU_TEMPORAL_DELIMITER &&
        obu->has_extension && p->have_seq && p->op_idc != 0) {
        const int in_t = (p->op_idc >> obu->temporal_id) & 1;
        const int in_s = (p->op_idc >> (obu->spatial_id + 8)) & 1;
        if (!in_t || !in_s) {
            obu->dropped = 1;
            return 0;
        }
    }
    switch (obu->type) {
    case AV1_OBU_SEQUENCE_HEADER: {
        if (pavail < obu->payload_len && pavail < 64)
            return -2;
        Av1SeqHdr s;
        if (av1_parse_sequence_header(pl, pavail < obu->payload_len ? pavail : obu->payload_len, &s, p->err,
                                      sizeof(p->err)) < 0)
            return pavail < obu->payload_len ? -2 : -1;
        p->seq = s;
        p->have_seq = 1;
        const int op = p->operating_point <= s.operating_points_cnt_minus_1 ? p->operating_point : 0;
        p->op_idc = s.operating_point_idc[op];
        return 0;
    }
    case AV1_OBU_TEMPORAL_DELIMITER:
        p->seen_frame_header = 0;
        return 0;
    case AV1_OBU_FRAME_HEADER:
    case AV1_OBU_REDUNDANT_FRAME_HEADER:
    case AV1_OBU_FRAME: {
        if (!p->have_seq) {
            seterr(p->err, sizeof(p->err), "frame header before the first sequence header");
            return -1;
        }
        /* uncompressed_header() changes the reference state as it goes
         * (order hints of invalid refs, key-frame resets): keep a copy so
         * that a failed or truncated parse leaves the state untouched and a
         * retry with more bytes starts from the same state */
        const Av1Parser save = *p; /* about 10 KB */
        int r = frame_header_obu(p, pl, pavail, obu->payload_len, obu, &b);
        if (r == 0 && obu->type == AV1_OBU_FRAME) {
            if (obu->frame_hdr.show_existing_frame) {
                seterr(p->err, sizeof(p->err), "OBU_FRAME with show_existing_frame");
                r = -1;
            } else {
                byte_alignment(&b);
                r = tile_group_obu(p, &b, obu);
            }
        }
        if (r < 0)
            *p = save;
        return r;
    }
    case AV1_OBU_TILE_GROUP: {
        const int t = tile_group_obu(p, &b, obu);
        return t;
    }
    default:
        return 0;
    }
}
