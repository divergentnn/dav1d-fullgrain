/*
 * Full-frame AR film grain: AVX2 kernel (included by fg_variant_tmpl.c).
 *
 * The AR recursion is causal: sample (r, c) needs (r, c-1..c-3) and the three
 * rows above at columns c-3..c+3. A plain raster scan is a scalar dependency
 * chain. Here 16 rows run in lockstep, one per int16 lane, skewed by SKEW
 * columns per row: at step t, lane j computes column c = t - SKEW*j of row
 * r0 + j. Stored in a skewed column-major layout,
 *     idx(r, c) = (c + SKEW*r) * Rp + r,
 * the 16 lanes of a step are contiguous and every tap (dy, dx) is a plain
 * vector load at the constant offset (dx + SKEW*dy) * Rp + dy. In-row taps come
 * from the previous three step vectors (registers); with SKEW = 8 every load
 * is >= 5 steps old, so no store-forwarding stalls. White noise is generated
 * inside the loop (same hash as noise_row), results are transposed 16x16 back
 * to a row-major field for the apply step and for the chroma-from-luma term.
 *
 * Rows 0..2 and columns 0..2 / bw-3..bw-1 are white noise (inactive lanes).
 * Integer arithmetic is identical to generate_grain_{y,uv}_c / ar_band().
 */

#define SKEW 8
#define SK_PADL 128   /* >= SKEW * 15 */
#define SK_PADR 160   /* >= SKEW * 15 + 15 + slack */

typedef struct SkewDims {
    int bw, R, Rp, T16;
    ptrdiff_t ostr;        /* row-major output stride */
    size_t skew_elems, out_elems;
    ptrdiff_t skew_base;   /* element offset of idx 0 inside the skew buffer */
} SkewDims;

static inline void skew_dims(SkewDims *const d, const int bw, const int R) {
    d->bw = bw;
    d->R = R;
    const int ng = (R + 15) >> 4;
    d->Rp = 16 * ng;
    d->T16 = (bw + SKEW * 15 + 15) & ~15;
    d->ostr = (SK_PADL + bw + SK_PADR + 15) & ~15;
    d->skew_base = (ptrdiff_t) (3 * SKEW + 4) * d->Rp + 16;
    d->skew_elems = (size_t) d->skew_base +
                    (size_t) (d->T16 + SKEW * d->Rp + 32) * d->Rp + 64;
    d->out_elems = (size_t) d->Rp * d->ostr + 64;
}

#if ARCH_X86_64
static inline FGV_AVX2 void tr8x8_lanes(__m256i a[8]) {
    __m256i b[8], c[8];
    for (int i = 0; i < 4; i++) {
        b[2 * i]     = _mm256_unpacklo_epi16(a[2 * i], a[2 * i + 1]);
        b[2 * i + 1] = _mm256_unpackhi_epi16(a[2 * i], a[2 * i + 1]);
    }
    c[0] = _mm256_unpacklo_epi32(b[0], b[2]);
    c[1] = _mm256_unpackhi_epi32(b[0], b[2]);
    c[2] = _mm256_unpacklo_epi32(b[1], b[3]);
    c[3] = _mm256_unpackhi_epi32(b[1], b[3]);
    c[4] = _mm256_unpacklo_epi32(b[4], b[6]);
    c[5] = _mm256_unpackhi_epi32(b[4], b[6]);
    c[6] = _mm256_unpacklo_epi32(b[5], b[7]);
    c[7] = _mm256_unpackhi_epi32(b[5], b[7]);
    a[0] = _mm256_unpacklo_epi64(c[0], c[4]);
    a[1] = _mm256_unpackhi_epi64(c[0], c[4]);
    a[2] = _mm256_unpacklo_epi64(c[1], c[5]);
    a[3] = _mm256_unpackhi_epi64(c[1], c[5]);
    a[4] = _mm256_unpacklo_epi64(c[2], c[6]);
    a[5] = _mm256_unpackhi_epi64(c[2], c[6]);
    a[6] = _mm256_unpacklo_epi64(c[3], c[7]);
    a[7] = _mm256_unpackhi_epi64(c[3], c[7]);
}

/* in-place 16x16 int16 transpose: v[k] becomes column k */
static inline FGV_AVX2 void transpose16(__m256i v[16]) {
    tr8x8_lanes(v);
    tr8x8_lanes(v + 8);
    for (int k = 0; k < 8; k++) {
        const __m256i d = v[k], e = v[8 + k];
        v[k] = _mm256_permute2x128_si256(d, e, 0x20);
        v[8 + k] = _mm256_permute2x128_si256(d, e, 0x31);
    }
}

/* v[s] = step t0 + s (lanes = rows r0..r0+15) -> row-major out */
static inline FGV_AVX2 void skew_transpose_store(__m256i v[16], int16_t *const out,
                                                 const ptrdiff_t ostr, const int r0,
                                                 const int t0)
{
    transpose16(v);
    for (int k = 0; k < 16; k++)
        _mm256_storeu_si256((__m256i *) (out + (ptrdiff_t) (r0 + k) * ostr + SK_PADL +
                                         t0 - SKEW * k), v[k]);
}

/* row-major in -> step vectors for lanes = rows r0..r0+15, steps t0..t0+15 */
static inline FGV_AVX2 void skew_transpose_load(__m256i v[16], const int16_t *const in,
                                                const ptrdiff_t istr, const int r0,
                                                const int t0)
{
    for (int k = 0; k < 16; k++)
        v[k] = _mm256_loadu_si256((const __m256i *) (in + (ptrdiff_t) (r0 + k) * istr +
                                                     SK_PADL + t0 - SKEW * k));
    transpose16(v);
}

/* chroma-from-luma term at chroma resolution, row-major with the skew
 * paddings: lt[r][c] = round2(sum of the co-located luma field samples) */
static NOINLINE FGV_AVX2 void lumaterm_rows_avx2(int16_t *const lt, const ptrdiff_t ltstr,
                                                 const int16_t *const lum,
                                                 const ptrdiff_t lstr, const int rows,
                                                 const int bw, const int subx,
                                                 const int suby)
{
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i rnd = _mm256_set1_epi32((1 << (subx + suby)) >> 1);
    const __m128i sh = _mm_cvtsi32_si128(subx + suby);
    const __m256i rnd16 = _mm256_set1_epi16((1 << suby) >> 1);
    for (int r = 0; r < rows; r++) {
        int16_t *const o = lt + (ptrdiff_t) r * ltstr + SK_PADL;
        const int16_t *const l0 = lum + (ptrdiff_t) (r << suby) * lstr;
        const int16_t *const l1 = l0 + (suby ? lstr : 0);
        int c = 0;
        if (subx) {
            for (; c + 16 <= bw; c += 16) {
                __m256i a = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) (l0 + 2 * c)), ones);
                __m256i b = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) (l0 + 2 * c + 16)), ones);
                if (suby) {
                    a = _mm256_add_epi32(a, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) (l1 + 2 * c)), ones));
                    b = _mm256_add_epi32(b, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) (l1 + 2 * c + 16)), ones));
                }
                a = _mm256_sra_epi32(_mm256_add_epi32(a, rnd), sh);
                b = _mm256_sra_epi32(_mm256_add_epi32(b, rnd), sh);
                _mm256_storeu_si256((__m256i *) (o + c),
                                    _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8));
            }
        } else if (suby) {
            for (; c + 16 <= bw; c += 16) {
                const __m256i a = _mm256_loadu_si256((const __m256i *) (l0 + c));
                const __m256i b = _mm256_loadu_si256((const __m256i *) (l1 + c));
                /* (a + b + 1) >> 1 without overflow: values are 12-bit */
                _mm256_storeu_si256((__m256i *) (o + c),
                    _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(a, b), rnd16), 1));
            }
        } else {
            for (; c + 16 <= bw; c += 16)
                _mm256_storeu_si256((__m256i *) (o + c),
                                    _mm256_loadu_si256((const __m256i *) (l0 + c)));
        }
        for (; c < bw; c++) {
            const int lc = c << subx;
            int luma = l0[lc];
            if (subx) luma += l0[lc + 1];
            if (suby) {
                luma += l1[lc];
                if (subx) luma += l1[lc + 1];
            }
            o[c] = (int16_t) ((luma + ((1 << (subx + suby)) >> 1)) >> (subx + suby));
        }
    }
}

static inline FGV_AVX2 __m256i fg_hash8(__m256i h) {
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
    h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x7feb352d));
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 15));
    h = _mm256_mullo_epi32(h, _mm256_set1_epi32((int) 0x846ca68bu));
    return _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
}

/*
 * One band of one plane. Band row r is absolute row y0 + r, column c is
 * absolute column x0 + c (x0 even). Rows 0..2 and columns 0..2 / bw-3..bw-1
 * stay white noise, the AR runs on the rest. Output: row-major field
 * out[r * ostr + SK_PADL + c] for r < R, c < bw.
 * Chroma: lt is the chroma-from-luma term (lumaterm_rows_avx2, same layout
 * and stride as out), added with weight luma_coeff when non-zero.
 */
static NOINLINE FGV_AVX2 void
ar_band_skew_avx2(int16_t *const out, int16_t *const skewbuf, const SkewDims *const d,
                  const uint32_t key, const int y0, const int x0,
                  const int32_t *const gtab, const int8_t *const coeffs, const int lag,
                  const int ar_shift, const int gmin, const int gmax,
                  const int16_t *const lt, const int luma_coeff)
{
    const int bw = d->bw, R = d->R, Rp = d->Rp;
    const ptrdiff_t ostr = d->ostr;
    int16_t *const sk = skewbuf + d->skew_base;
    const uint32_t KX = 0x85ebca6bu;

    /* taps: previous rows from memory, in-row from registers */
    ptrdiff_t moff[24];
    int mcf[24], nm = 0;
    {
        const int8_t *co = coeffs;
        for (int dy = -lag; dy < 0; dy++)
            for (int dx = -lag; dx <= lag; dx++) {
                const int cf = *(co++);
                if (cf) { moff[nm] = (ptrdiff_t) (dx + SKEW * dy) * Rp + dy; mcf[nm] = cf; nm++; }
            }
    }
    const int8_t *const cin = coeffs + lag * (2 * lag + 1);
    const int c3 = lag >= 3 ? cin[lag - 3] : 0;
    const int c2 = lag >= 2 ? cin[lag - 2] : 0;
    const int c1 = lag >= 1 ? cin[lag - 1] : 0;
#define PACK2(a, b) _mm256_set1_epi32((int) ((uint32_t) (uint16_t) (a) | ((uint32_t) (uint16_t) (b) << 16)))
    const __m256i cv12 = PACK2(c1, c2);
    const __m256i cv3m = PACK2(c3, nm ? mcf[0] : 0);
    __m256i mcv[12];
    ptrdiff_t mo2[24];
    int np = 0;
    for (int k = 1; k < nm; k += 2) {
        mo2[2 * np] = moff[k];
        mo2[2 * np + 1] = k + 1 < nm ? moff[k + 1] : 0;
        mcv[np] = PACK2(mcf[k], k + 1 < nm ? mcf[k + 1] : 0);
        np++;
    }
#undef PACK2
    const ptrdiff_t m0off = nm ? moff[0] : 0;
    const __m256i rnd = _mm256_set1_epi32((1 << ar_shift) >> 1);
    const __m128i shv = _mm_cvtsi32_si128(ar_shift);
    const __m256i vgmin = _mm256_set1_epi16(gmin), vgmax = _mm256_set1_epi16(gmax);
    const __m256i lane16 = _mm256_setr_epi16(0, -SKEW, -2 * SKEW, -3 * SKEW, -4 * SKEW, -5 * SKEW,
                                             -6 * SKEW, -7 * SKEW, -8 * SKEW, -9 * SKEW, -10 * SKEW,
                                             -11 * SKEW, -12 * SKEW, -13 * SKEW, -14 * SKEW,
                                             -15 * SKEW);
    const __m256i lo3 = _mm256_set1_epi16(2), hiw = _mm256_set1_epi16(bw - 3);
    const __m256i m11 = _mm256_set1_epi32(0x7ff);
    const int has_luma = lt && luma_coeff;
    const __m256i lcv = _mm256_set1_epi32((int) ((uint32_t) (uint16_t) luma_coeff));
    const __m256i zero = _mm256_setzero_si256();

    /* rows 0..2 are white noise: lanes 0..2 of the first group are inactive */
    const __m256i lanerow = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (int r0 = 0; r0 < R; r0 += 16) {
        const __m256i rowact = _mm256_cmpgt_epi16(_mm256_add_epi16(lanerow, _mm256_set1_epi16(r0)), lo3);
        /* per-lane noise keys: h_in = B_j + q * K, q = (x0 + t) >> 1 */
        uint32_t Bv[16];
        for (int j = 0; j < 16; j++) {
            const uint32_t rk = dav1d_fgv_hash(key ^ dav1d_fgv_hash((uint32_t) (y0 + r0 + j) *
                                                                    0x9E3779B9u + 0x7f4a7c15u));
            Bv[j] = rk - (uint32_t) (SKEW / 2 * j) * KX;
        }
        const __m256i B0 = _mm256_loadu_si256((const __m256i *) Bv);
        const __m256i B1 = _mm256_loadu_si256((const __m256i *) (Bv + 8));
        int16_t *const P0 = sk + (ptrdiff_t) SKEW * r0 * Rp + r0;
        __m256i V1 = _mm256_setzero_si256(), V2 = V1, V3 = V1;
        __m256i We = V1, Wo = V1;
        __m256i vbuf[16], ltv[16];
        for (int t0 = 0; t0 < d->T16; t0 += 16) {
            if (has_luma) skew_transpose_load(ltv, lt, ostr, r0, t0);
            for (int s = 0; s < 16; s++) {
                const int t = t0 + s;
                int16_t *const P = P0 + (ptrdiff_t) t * Rp;
                if (!(t & 1)) {
                    const __m256i qK = _mm256_set1_epi32((int) ((uint32_t) ((x0 + t) >> 1) * KX));
                    const __m256i h0 = fg_hash8(_mm256_add_epi32(B0, qK));
                    const __m256i h1 = fg_hash8(_mm256_add_epi32(B1, qK));
                    const __m256i e0 = _mm256_i32gather_epi32((const int *) gtab,
                                           _mm256_and_si256(_mm256_srli_epi32(h0, 5), m11), 4);
                    const __m256i e1 = _mm256_i32gather_epi32((const int *) gtab,
                                           _mm256_and_si256(_mm256_srli_epi32(h1, 5), m11), 4);
                    const __m256i o0 = _mm256_i32gather_epi32((const int *) gtab,
                                           _mm256_srli_epi32(h0, 21), 4);
                    const __m256i o1 = _mm256_i32gather_epi32((const int *) gtab,
                                           _mm256_srli_epi32(h1, 21), 4);
                    We = _mm256_permute4x64_epi64(_mm256_packs_epi32(e0, e1), 0xD8);
                    Wo = _mm256_permute4x64_epi64(_mm256_packs_epi32(o0, o1), 0xD8);
                }
                const __m256i W = (t & 1) ? Wo : We;
                /* memory taps and chroma-from-luma first (independent of the
                 * previous step), in-row taps last: keeps the loop-carried
                 * dependency chain short */
                __m256i lo = rnd, hi = rnd;
                for (int k = 0; k < np; k++) {
                    const __m256i a = _mm256_loadu_si256((const __m256i *) (P + mo2[2 * k]));
                    const __m256i b = _mm256_loadu_si256((const __m256i *) (P + mo2[2 * k + 1]));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), mcv[k]));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), mcv[k]));
                }
                if (has_luma) {
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(ltv[s], zero), lcv));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(ltv[s], zero), lcv));
                }
                {
                    const __m256i m = _mm256_loadu_si256((const __m256i *) (P + m0off));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(V3, m), cv3m));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(V3, m), cv3m));
                }
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(V1, V2), cv12));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(V1, V2), cv12));
                lo = _mm256_sra_epi32(lo, shv);
                hi = _mm256_sra_epi32(hi, shv);
                __m256i v = _mm256_adds_epi16(W, _mm256_packs_epi32(lo, hi));
                v = _mm256_min_epi16(_mm256_max_epi16(v, vgmin), vgmax);
                /* AR only on 3 <= c < bw - 3, white noise elsewhere */
                if (r0 < 3 || t < 3 + 15 * SKEW || t >= bw - 3) {
                    const __m256i cvec = _mm256_add_epi16(_mm256_set1_epi16(t), lane16);
                    const __m256i act = _mm256_and_si256(rowact, _mm256_and_si256(
                        _mm256_cmpgt_epi16(cvec, lo3), _mm256_cmpgt_epi16(hiw, cvec)));
                    v = _mm256_blendv_epi8(W, v, act);
                }
                _mm256_storeu_si256((__m256i *) P, v);
                vbuf[s] = v;
                V3 = V2; V2 = V1; V1 = v;
            }
            skew_transpose_store(vbuf, out, ostr, r0, t0);
        }
    }
}
#elif ARCH_AARCH64
/*
 * NEON version: same skewed layout, same 16 lanes (two int16x8 registers,
 * rows r0..r0+7 and r0+8..r0+15), same arithmetic. Taps use widening
 * multiply-accumulate (vmlal), one tap at a time; noise for 16 steps is
 * generated per chunk (scalar table reads: NEON has no gather).
 */
static inline void tr8x8_neon(int16x8_t *const r) {
    const int16x8_t t0 = vtrn1q_s16(r[0], r[1]), t1 = vtrn2q_s16(r[0], r[1]);
    const int16x8_t t2 = vtrn1q_s16(r[2], r[3]), t3 = vtrn2q_s16(r[2], r[3]);
    const int16x8_t t4 = vtrn1q_s16(r[4], r[5]), t5 = vtrn2q_s16(r[4], r[5]);
    const int16x8_t t6 = vtrn1q_s16(r[6], r[7]), t7 = vtrn2q_s16(r[6], r[7]);
#define S32(x) vreinterpretq_s32_s16(x)
    const int32x4_t u0 = vtrn1q_s32(S32(t0), S32(t2)), u2 = vtrn2q_s32(S32(t0), S32(t2));
    const int32x4_t u1 = vtrn1q_s32(S32(t1), S32(t3)), u3 = vtrn2q_s32(S32(t1), S32(t3));
    const int32x4_t u4 = vtrn1q_s32(S32(t4), S32(t6)), u6 = vtrn2q_s32(S32(t4), S32(t6));
    const int32x4_t u5 = vtrn1q_s32(S32(t5), S32(t7)), u7 = vtrn2q_s32(S32(t5), S32(t7));
#undef S32
#define S64(x) vreinterpretq_s64_s32(x)
#define R16(x) vreinterpretq_s16_s64(x)
    r[0] = R16(vtrn1q_s64(S64(u0), S64(u4))); r[4] = R16(vtrn2q_s64(S64(u0), S64(u4)));
    r[1] = R16(vtrn1q_s64(S64(u1), S64(u5))); r[5] = R16(vtrn2q_s64(S64(u1), S64(u5)));
    r[2] = R16(vtrn1q_s64(S64(u2), S64(u6))); r[6] = R16(vtrn2q_s64(S64(u2), S64(u6)));
    r[3] = R16(vtrn1q_s64(S64(u3), S64(u7))); r[7] = R16(vtrn2q_s64(S64(u3), S64(u7)));
#undef S64
#undef R16
}

/* a[s] = step t0+s, lanes 0-7 (rows r0..r0+7); b[s] = lanes 8-15 */
static inline void skew_transpose_store_neon(int16x8_t *const a, int16x8_t *const b,
                                             int16_t *const out, const ptrdiff_t ostr,
                                             const int r0, const int t0)
{
    tr8x8_neon(a); tr8x8_neon(a + 8); tr8x8_neon(b); tr8x8_neon(b + 8);
    for (int k = 0; k < 8; k++) {
        int16_t *const o0 = out + (ptrdiff_t) (r0 + k) * ostr + SK_PADL + t0 - SKEW * k;
        int16_t *const o1 = out + (ptrdiff_t) (r0 + 8 + k) * ostr + SK_PADL + t0 - SKEW * (8 + k);
        vst1q_s16(o0, a[k]); vst1q_s16(o0 + 8, a[8 + k]);
        vst1q_s16(o1, b[k]); vst1q_s16(o1 + 8, b[8 + k]);
    }
}

/* row-major in -> step vectors a[s] (lanes 0-7), b[s] (lanes 8-15) */
static inline void skew_transpose_load_neon(int16x8_t *const a, int16x8_t *const b,
                                            const int16_t *const in, const ptrdiff_t istr,
                                            const int r0, const int t0)
{
    for (int k = 0; k < 8; k++) {
        const int16_t *const i0 = in + (ptrdiff_t) (r0 + k) * istr + SK_PADL + t0 - SKEW * k;
        const int16_t *const i1 = in + (ptrdiff_t) (r0 + 8 + k) * istr + SK_PADL + t0 - SKEW * (8 + k);
        a[k] = vld1q_s16(i0); a[8 + k] = vld1q_s16(i0 + 8);
        b[k] = vld1q_s16(i1); b[8 + k] = vld1q_s16(i1 + 8);
    }
    tr8x8_neon(a); tr8x8_neon(a + 8); tr8x8_neon(b); tr8x8_neon(b + 8);
}

static NOINLINE void lumaterm_rows_neon(int16_t *const lt, const ptrdiff_t ltstr,
                                        const int16_t *const lum, const ptrdiff_t lstr,
                                        const int rows, const int bw, const int subx,
                                        const int suby)
{
    const int16x8_t nsh = vdupq_n_s16(-(subx + suby));
    for (int r = 0; r < rows; r++) {
        int16_t *const o = lt + (ptrdiff_t) r * ltstr + SK_PADL;
        const int16_t *const l0 = lum + (ptrdiff_t) (r << suby) * lstr;
        const int16_t *const l1 = l0 + (suby ? lstr : 0);
        int c = 0;
        for (; c + 8 <= bw; c += 8) {
            int16x8_t v;
            if (subx) {
                v = vpaddq_s16(vld1q_s16(l0 + 2 * c), vld1q_s16(l0 + 2 * c + 8));
                if (suby) v = vaddq_s16(v, vpaddq_s16(vld1q_s16(l1 + 2 * c), vld1q_s16(l1 + 2 * c + 8)));
            } else {
                v = vld1q_s16(l0 + c);
                if (suby) v = vaddq_s16(v, vld1q_s16(l1 + c));
            }
            vst1q_s16(o + c, vrshlq_s16(v, nsh));
        }
        for (; c < bw; c++) {
            const int lc = c << subx;
            int luma = l0[lc];
            if (subx) luma += l0[lc + 1];
            if (suby) {
                luma += l1[lc];
                if (subx) luma += l1[lc + 1];
            }
            o[c] = (int16_t) ((luma + ((1 << (subx + suby)) >> 1)) >> (subx + suby));
        }
    }
}

static inline uint32x4_t fg_hash4(uint32x4_t h) {
    h = veorq_u32(h, vshrq_n_u32(h, 16));
    h = vmulq_n_u32(h, 0x7feb352du);
    h = veorq_u32(h, vshrq_n_u32(h, 15));
    h = vmulq_n_u32(h, 0x846ca68bu);
    return veorq_u32(h, vshrq_n_u32(h, 16));
}

static NOINLINE void
ar_band_skew_neon(int16_t *const out, int16_t *const skewbuf, const SkewDims *const d,
                  const uint32_t key, const int y0, const int x0,
                  const int32_t *const gtab, const int8_t *const coeffs, const int lag,
                  const int ar_shift, const int gmin, const int gmax,
                  const int16_t *const lt, const int luma_coeff)
{
    const int bw = d->bw, R = d->R, Rp = d->Rp;
    const ptrdiff_t ostr = d->ostr;
    int16_t *const sk = skewbuf + d->skew_base;
    const uint32_t KX = 0x85ebca6bu;

    ptrdiff_t moff[24];
    int16_t mcf[24];
    int nm = 0;
    {
        const int8_t *co = coeffs;
        for (int dy = -lag; dy < 0; dy++)
            for (int dx = -lag; dx <= lag; dx++) {
                const int cf = *(co++);
                if (cf) { moff[nm] = (ptrdiff_t) (dx + SKEW * dy) * Rp + dy; mcf[nm] = cf; nm++; }
            }
    }
    const int8_t *const cin = coeffs + lag * (2 * lag + 1);
    const int16_t c3 = lag >= 3 ? cin[lag - 3] : 0;
    const int16_t c2 = lag >= 2 ? cin[lag - 2] : 0;
    const int16_t c1 = lag >= 1 ? cin[lag - 1] : 0;
    const int16_t lc = (int16_t) luma_coeff;
    const int has_luma = lt && luma_coeff;
    const int32x4_t rnd = vdupq_n_s32((1 << ar_shift) >> 1);
    const int32x4_t nsh = vdupq_n_s32(-ar_shift);
    const int16x8_t vgmin = vdupq_n_s16(gmin), vgmax = vdupq_n_s16(gmax);
    const int16_t laneA_[8] = { 0, -SKEW, -2 * SKEW, -3 * SKEW, -4 * SKEW, -5 * SKEW, -6 * SKEW, -7 * SKEW };
    const int16x8_t laneA = vld1q_s16(laneA_);
    const int16x8_t laneB = vaddq_s16(laneA, vdupq_n_s16(-8 * SKEW));
    const int16_t rowsA_[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const int16x8_t rowA = vld1q_s16(rowsA_), rowB = vaddq_s16(rowA, vdupq_n_s16(8));
    const int16x8_t lo3 = vdupq_n_s16(2), hiw = vdupq_n_s16(bw - 3);
    const uint32_t lanes4_[4] = { 0, 1, 2, 3 };
    (void) lanes4_;

    for (int r0 = 0; r0 < R; r0 += 16) {
        const uint16x8_t rowactA = vcgtq_s16(vaddq_s16(rowA, vdupq_n_s16(r0)), lo3);
        const uint16x8_t rowactB = vcgtq_s16(vaddq_s16(rowB, vdupq_n_s16(r0)), lo3);
        uint32_t Bv[16];
        for (int j = 0; j < 16; j++) {
            const uint32_t rk = dav1d_fgv_hash(key ^ dav1d_fgv_hash((uint32_t) (y0 + r0 + j) *
                                                                    0x9E3779B9u + 0x7f4a7c15u));
            Bv[j] = rk - (uint32_t) (SKEW / 2 * j) * KX;
        }
        const uint32x4_t Bq0 = vld1q_u32(Bv), Bq1 = vld1q_u32(Bv + 4);
        const uint32x4_t Bq2 = vld1q_u32(Bv + 8), Bq3 = vld1q_u32(Bv + 12);
        int16_t *const P0 = sk + (ptrdiff_t) SKEW * r0 * Rp + r0;
        int16x8_t V1a = vdupq_n_s16(0), V2a = V1a, V3a = V1a;
        int16x8_t V1b = V1a, V2b = V1a, V3b = V1a;
        int16x8_t va[16], vb[16], la[16], lb[16];
        int16_t wbuf[16][16];
        for (int t0 = 0; t0 < d->T16; t0 += 16) {
            if (has_luma) skew_transpose_load_neon(la, lb, lt, ostr, r0, t0);
            /* white noise for 16 steps x 16 lanes (one hash per sample pair) */
            for (int p = 0; p < 8; p++) {
                const uint32_t qK = (uint32_t) ((x0 + t0 + 2 * p) >> 1) * KX;
                uint32_t hv[16];
                vst1q_u32(hv,      fg_hash4(vaddq_u32(Bq0, vdupq_n_u32(qK))));
                vst1q_u32(hv + 4,  fg_hash4(vaddq_u32(Bq1, vdupq_n_u32(qK))));
                vst1q_u32(hv + 8,  fg_hash4(vaddq_u32(Bq2, vdupq_n_u32(qK))));
                vst1q_u32(hv + 12, fg_hash4(vaddq_u32(Bq3, vdupq_n_u32(qK))));
                for (int j = 0; j < 16; j++) {
                    wbuf[2 * p][j] = (int16_t) gtab[(hv[j] >> 5) & 0x7ff];
                    wbuf[2 * p + 1][j] = (int16_t) gtab[hv[j] >> 21];
                }
            }
            for (int s = 0; s < 16; s++) {
                const int t = t0 + s;
                int16_t *const P = P0 + (ptrdiff_t) t * Rp;
                const int16x8_t WA = vld1q_s16(wbuf[s]), WB = vld1q_s16(wbuf[s] + 8);
                int32x4_t aAl = rnd, aAh = rnd, aBl = rnd, aBh = rnd;
                for (int k = 0; k < nm; k++) {
                    const int16x8_t a = vld1q_s16(P + moff[k]);
                    const int16x8_t b = vld1q_s16(P + moff[k] + 8);
                    aAl = vmlal_n_s16(aAl, vget_low_s16(a), mcf[k]);
                    aAh = vmlal_high_n_s16(aAh, a, mcf[k]);
                    aBl = vmlal_n_s16(aBl, vget_low_s16(b), mcf[k]);
                    aBh = vmlal_high_n_s16(aBh, b, mcf[k]);
                }
                if (has_luma) {
                    aAl = vmlal_n_s16(aAl, vget_low_s16(la[s]), lc);
                    aAh = vmlal_high_n_s16(aAh, la[s], lc);
                    aBl = vmlal_n_s16(aBl, vget_low_s16(lb[s]), lc);
                    aBh = vmlal_high_n_s16(aBh, lb[s], lc);
                }
                aAl = vmlal_n_s16(aAl, vget_low_s16(V3a), c3);
                aAh = vmlal_high_n_s16(aAh, V3a, c3);
                aBl = vmlal_n_s16(aBl, vget_low_s16(V3b), c3);
                aBh = vmlal_high_n_s16(aBh, V3b, c3);
                aAl = vmlal_n_s16(aAl, vget_low_s16(V2a), c2);
                aAh = vmlal_high_n_s16(aAh, V2a, c2);
                aBl = vmlal_n_s16(aBl, vget_low_s16(V2b), c2);
                aBh = vmlal_high_n_s16(aBh, V2b, c2);
                aAl = vmlal_n_s16(aAl, vget_low_s16(V1a), c1);
                aAh = vmlal_high_n_s16(aAh, V1a, c1);
                aBl = vmlal_n_s16(aBl, vget_low_s16(V1b), c1);
                aBh = vmlal_high_n_s16(aBh, V1b, c1);
                const int16x8_t sA = vcombine_s16(vqmovn_s32(vshlq_s32(aAl, nsh)),
                                                  vqmovn_s32(vshlq_s32(aAh, nsh)));
                const int16x8_t sB = vcombine_s16(vqmovn_s32(vshlq_s32(aBl, nsh)),
                                                  vqmovn_s32(vshlq_s32(aBh, nsh)));
                int16x8_t vA = vminq_s16(vmaxq_s16(vqaddq_s16(WA, sA), vgmin), vgmax);
                int16x8_t vB = vminq_s16(vmaxq_s16(vqaddq_s16(WB, sB), vgmin), vgmax);
                if (r0 < 3 || t < 3 + 15 * SKEW || t >= bw - 3) {
                    const int16x8_t tv = vdupq_n_s16(t);
                    const int16x8_t cA = vaddq_s16(tv, laneA), cB = vaddq_s16(tv, laneB);
                    const uint16x8_t actA = vandq_u16(rowactA, vandq_u16(vcgtq_s16(cA, lo3),
                                                                        vcgtq_s16(hiw, cA)));
                    const uint16x8_t actB = vandq_u16(rowactB, vandq_u16(vcgtq_s16(cB, lo3),
                                                                        vcgtq_s16(hiw, cB)));
                    vA = vbslq_s16(actA, vA, WA);
                    vB = vbslq_s16(actB, vB, WB);
                }
                vst1q_s16(P, vA);
                vst1q_s16(P + 8, vB);
                va[s] = vA; vb[s] = vB;
                V3a = V2a; V2a = V1a; V1a = vA;
                V3b = V2b; V2b = V1b; V1b = vB;
            }
            skew_transpose_store_neon(va, vb, out, ostr, r0, t0);
        }
    }
}
#endif
