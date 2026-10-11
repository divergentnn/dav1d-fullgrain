/*
 * av1hdr: a small, standalone AV1 header parser (no pixel decoding).
 *
 * It parses OBU headers, sequence headers and uncompressed frame headers far
 * enough to reach film_grain_params(), keeping the reference state the
 * frame header syntax depends on (order hints, frame sizes, segmentation and
 * loop-filter deltas, global motion, film grain per reference slot). For
 * every frame header it reports where film_grain_params() and grain_seed sit
 * in the OBU payload, and the effective grain parameters.
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#ifndef AV1HDR_H
#define AV1HDR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    AV1_OBU_SEQUENCE_HEADER = 1,
    AV1_OBU_TEMPORAL_DELIMITER = 2,
    AV1_OBU_FRAME_HEADER = 3,
    AV1_OBU_TILE_GROUP = 4,
    AV1_OBU_METADATA = 5,
    AV1_OBU_FRAME = 6,
    AV1_OBU_REDUNDANT_FRAME_HEADER = 7,
    AV1_OBU_TILE_LIST = 8,
    AV1_OBU_PADDING = 15,
};

enum { AV1_KEY_FRAME = 0, AV1_INTER_FRAME = 1, AV1_INTRA_ONLY_FRAME = 2, AV1_SWITCH_FRAME = 3 };

/* film_grain_params() syntax elements, as coded (the *_plus_128 / offsets
 * are stored raw; subtract 128 / 256 for the signed values). */
typedef struct Av1FilmGrain {
    uint8_t apply_grain;
    uint16_t grain_seed;
    uint8_t update_grain;               /* value after load_grain_params() */
    uint8_t film_grain_params_ref_idx;
    uint8_t num_y_points;
    uint8_t point_y_value[16], point_y_scaling[16];
    uint8_t chroma_scaling_from_luma;
    uint8_t num_cb_points;
    uint8_t point_cb_value[16], point_cb_scaling[16];
    uint8_t num_cr_points;
    uint8_t point_cr_value[16], point_cr_scaling[16];
    uint8_t grain_scaling_minus_8;
    uint8_t ar_coeff_lag;
    uint8_t ar_coeffs_y_plus_128[24];
    uint8_t ar_coeffs_cb_plus_128[25];
    uint8_t ar_coeffs_cr_plus_128[25];
    uint8_t ar_coeff_shift_minus_6;
    uint8_t grain_scale_shift;
    uint8_t cb_mult, cb_luma_mult;
    uint16_t cb_offset;
    uint8_t cr_mult, cr_luma_mult;
    uint16_t cr_offset;
    uint8_t overlap_flag;
    uint8_t clip_to_restricted_range;
} Av1FilmGrain;

typedef struct Av1SeqHdr {
    int seq_profile, still_picture, reduced_still_picture_header;
    int timing_info_present_flag, decoder_model_info_present_flag;
    int equal_picture_interval;
    uint32_t num_units_in_display_tick, time_scale, num_ticks_per_picture_minus_1;
    int buffer_delay_length_minus_1, buffer_removal_time_length_minus_1;
    int frame_presentation_time_length_minus_1;
    int initial_display_delay_present_flag;
    int operating_points_cnt_minus_1;
    int operating_point_idc[32];
    int seq_level_idx[32], seq_tier[32];
    int decoder_model_present_for_this_op[32];
    int frame_width_bits_minus_1, frame_height_bits_minus_1;
    int max_frame_width_minus_1, max_frame_height_minus_1;
    int frame_id_numbers_present_flag;
    int delta_frame_id_length_minus_2, additional_frame_id_length_minus_1;
    int use_128x128_superblock;
    int enable_filter_intra, enable_intra_edge_filter;
    int enable_interintra_compound, enable_masked_compound;
    int enable_warped_motion, enable_dual_filter;
    int enable_order_hint, enable_jnt_comp, enable_ref_frame_mvs;
    int seq_force_screen_content_tools, seq_force_integer_mv;
    int order_hint_bits;
    int enable_superres, enable_cdef, enable_restoration;
    int bit_depth, mono_chrome, num_planes;
    int color_primaries, transfer_characteristics, matrix_coefficients;
    int color_range, subsampling_x, subsampling_y, chroma_sample_position;
    int separate_uv_delta_q;
    int film_grain_params_present;
} Av1SeqHdr;

/* One parsed frame header (OBU_FRAME_HEADER, OBU_FRAME or
 * OBU_REDUNDANT_FRAME_HEADER). Bit offsets count from the first bit of the
 * OBU payload (after the OBU header and its size field). */
typedef struct Av1FrameHdr {
    int obu_type;
    int temporal_id, spatial_id;
    int is_copy;               /* a frame_header_copy() of the current frame's header */
    int show_existing_frame, frame_to_show_map_idx;
    int frame_type, show_frame, showable_frame, error_resilient_mode;
    int disable_cdf_update, allow_screen_content_tools, force_integer_mv;
    int current_frame_id, frame_size_override_flag;
    int order_hint, primary_ref_frame, refresh_frame_flags;
    int ref_frame_idx[7];
    int frame_width, frame_height, upscaled_width;
    int render_width, render_height, use_superres, superres_denom;
    int allow_intrabc, allow_high_precision_mv;
    int is_filter_switchable, interpolation_filter, is_motion_mode_switchable;
    int use_ref_frame_mvs, disable_frame_end_update_cdf;
    int tile_cols, tile_rows, tile_cols_log2, tile_rows_log2;
    int context_update_tile_id, tile_size_bytes;
    int base_q_idx, delta_q_y_dc, delta_q_u_dc, delta_q_u_ac, delta_q_v_dc, delta_q_v_ac;
    int using_qmatrix;
    int segmentation_enabled;
    int delta_q_present, delta_lf_present;
    int coded_lossless, all_lossless;
    int loop_filter_level[4];
    int cdef_bits, tx_mode_select, reference_select, skip_mode_present;
    int allow_warped_motion, reduced_tx_set;
    int gm_type[8];
    int32_t gm_params[8][6];

    /* film grain */
    int fg_parsed;             /* film_grain_params() was read (grain present, shown or showable) */
    int64_t fg_bit_offset;     /* apply_grain, -1 if film_grain_params() is not in this header */
    int64_t fg_bit_end;        /* first bit after film_grain_params() */
    int64_t seed_bit_offset;   /* grain_seed, -1 if apply_grain == 0 or absent */
    int update_grain_coded;    /* update_grain as coded (1 for key / intra / switch frames) */
    Av1FilmGrain fg;           /* effective parameters (loaded from the reference if update_grain == 0;
                                  for show_existing_frame: the shown frame's stored parameters) */
    int64_t header_bits;       /* length of uncompressed_header() in bits */
} Av1FrameHdr;

typedef struct Av1Obu {
    int type;
    int has_extension, temporal_id, spatial_id;
    int has_size_field;
    size_t header_len;         /* OBU header + extension + size field, in bytes */
    size_t payload_len;
    size_t total_len;          /* header_len + payload_len */
    int dropped;               /* not part of the selected operating point (not parsed) */
    int has_frame_hdr;         /* frame_hdr below is valid */
    int frame_end;             /* this OBU completed a frame (last tile group, or show_existing_frame) */
    Av1FrameHdr frame_hdr;
} Av1Obu;

typedef struct Av1RefSlot {
    int valid, frame_id, frame_type, order_hint, showable;
    int upscaled_width, frame_width, frame_height, render_width, render_height;
    int saved_order_hints[8];
    int32_t gm_params[8][6];
    int8_t loop_filter_ref_deltas[8], loop_filter_mode_deltas[2];
    uint8_t feature_enabled[8][8];
    int16_t feature_data[8][8];
    Av1FilmGrain fg;
    int bit_depth, subsampling_x, subsampling_y;
} Av1RefSlot;

typedef struct Av1Parser {
    int have_seq;
    Av1SeqHdr seq;
    int operating_point;       /* selected operating point (default 0, as in dav1d) */
    int op_idc;
    Av1RefSlot ref[8];
    int seen_frame_header;
    Av1FrameHdr cur;           /* header of the frame being decoded */
    size_t cur_hdr_bytes;      /* bytes of its payload that hold uncompressed_header() */
    uint8_t cur_hdr_copy[1024];/* first bytes of the current header, to check copies */
    size_t cur_hdr_copy_len;
    int tiles_seen;
    /* current frame state, saved to the slots by the reference update process */
    int8_t lf_ref_deltas[8], lf_mode_deltas[2];
    uint8_t feature_enabled[8][8];
    int16_t feature_data[8][8];
    int32_t prev_gm_params[8][6];
    int order_hints[8];
    int strict;                /* refuse non-conformant streams (e.g. bad copies) */
    char err[200];
} Av1Parser;

void av1_parser_init(Av1Parser *p);

/* leb128() at buf; returns the number of bytes used, 0 on error. */
int av1_leb128(const uint8_t *buf, size_t len, uint64_t *value);

/* Read the OBU header (and size) at buf. size_hint is the number of bytes
 * available for this OBU when obu_has_size_field is 0 (sample or annexb
 * obu_length). Returns 0 on success, -1 on error (p->err). Does not parse the
 * payload. */
int av1_obu_header(Av1Parser *p, const uint8_t *buf, size_t len, size_t size_hint, Av1Obu *obu);

/* Parse one OBU. buf must hold the OBU header and enough of its payload for
 * the headers (a few hundred bytes; at most the whole OBU). avail is the number
 * of bytes in buf; size_hint as above. Returns 0 on success, -1 on error,
 * -2 if more bytes are needed. */
int av1_parse_obu(Av1Parser *p, const uint8_t *buf, size_t avail, size_t size_hint, Av1Obu *obu);

/* Parse a sequence_header_obu() payload into seq. Returns 0 / -1. */
int av1_parse_sequence_header(const uint8_t *buf, size_t len, Av1SeqHdr *seq, char *err, size_t errlen);

/* Write the 16-bit value into buf at bit offset bitpos (MSB first). */
void av1_put_bits16(uint8_t *buf, int64_t bitpos, unsigned value);
unsigned av1_get_bits16(const uint8_t *buf, int64_t bitpos);

#ifdef __cplusplus
}
#endif
#endif
