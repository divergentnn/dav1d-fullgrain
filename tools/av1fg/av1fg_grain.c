/*
 * av1fg_grain: AV1 film grain templates and block offsets (spec 7.18.3).
 *
 * Copyright © 2026, divergentnn (av1-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 */
#include "av1fg_grain.h"

#include <string.h>

static inline unsigned get_random_number(unsigned *reg, int bits)
{
    unsigned r = *reg;
    const unsigned bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    r = (r >> 1) | (bit << 15);
    *reg = r;
    return (r >> (16 - bits)) & ((1u << bits) - 1);
}

static inline int round2(int x, int n)
{
    return n ? (x + (1 << (n - 1))) >> n : x;
}

static inline int clip3(int lo, int hi, int x)
{
    return x < lo ? lo : x > hi ? hi : x;
}

void av1fg_white_luma(unsigned seed, int16_t w[AV1FG_TPL_H][AV1FG_TPL_W])
{
    unsigned reg = seed;
    for (int y = 0; y < AV1FG_TPL_H; y++)
        for (int x = 0; x < AV1FG_TPL_W; x++)
            w[y][x] = av1fg_gaussian_sequence[get_random_number(&reg, 11)];
}

void av1fg_templates(const Av1fgGrainShape *s, unsigned seed, int planes, Av1fgTemplates *t)
{
    const Av1FilmGrain *g = &s->fg;
    const int shift = 12 - s->bit_depth + g->grain_scale_shift;
    const int center = 128 << (s->bit_depth - 8);
    const int gmin = -center, gmax = (256 << (s->bit_depth - 8)) - 1 - center;
    const int ar_shift = g->ar_coeff_shift_minus_6 + 6;
    const int lag = g->ar_coeff_lag;
    const int need_luma = (planes & 1) || ((planes & 6) && g->num_y_points);

    if (need_luma) {
        unsigned reg = seed;
        for (int y = 0; y < AV1FG_TPL_H; y++)
            for (int x = 0; x < AV1FG_TPL_W; x++)
                t->y[y][x] = (int16_t)(g->num_y_points
                                           ? round2(av1fg_gaussian_sequence[get_random_number(&reg, 11)], shift)
                                           : 0);
        if (g->num_y_points) {
            int coef[24], dy[24], dx[24], n = 0;
            for (int r = -lag; r <= 0; r++)
                for (int c = -lag; c <= lag; c++) {
                    if (!r && !c)
                        break;
                    coef[n] = g->ar_coeffs_y_plus_128[n] - 128;
                    dy[n] = r;
                    dx[n] = c;
                    n++;
                }
            for (int y = 3; y < AV1FG_TPL_H; y++)
                for (int x = 3; x < AV1FG_TPL_W - 3; x++) {
                    int sum = 0;
                    for (int k = 0; k < n; k++)
                        sum += t->y[y + dy[k]][x + dx[k]] * coef[k];
                    t->y[y][x] = (int16_t)clip3(gmin, gmax, t->y[y][x] + round2(sum, ar_shift));
                }
        }
    }
    if (!(planes & 6) || s->mono)
        return;
    const int cw = s->ssx ? 44 : 82, ch = s->ssy ? 38 : 73;
    t->cw = cw;
    t->ch = ch;
    const int use_cb = g->num_cb_points || g->chroma_scaling_from_luma;
    const int use_cr = g->num_cr_points || g->chroma_scaling_from_luma;
    for (int pl = 0; pl < 2; pl++) {
        int16_t(*c)[AV1FG_TPL_W] = pl ? t->cr : t->cb;
        const int use = pl ? use_cr : use_cb;
        unsigned reg = seed ^ (pl ? 0x49d8 : 0xb524);
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++)
                c[y][x] = (int16_t)(use ? round2(av1fg_gaussian_sequence[get_random_number(&reg, 11)], shift) : 0);
    }
    int ccb[25], ccr[25], dy[25], dx[25], n = 0;
    for (int r = -lag; r <= 0; r++)
        for (int c = -lag; c <= lag; c++) {
            ccb[n] = g->ar_coeffs_cb_plus_128[n] - 128;
            ccr[n] = g->ar_coeffs_cr_plus_128[n] - 128;
            dy[n] = r;
            dx[n] = c;
            n++;
            if (!r && !c)
                break;
        }
    /* n - 1 neighbour taps, then the luma tap (index n - 1) */
    for (int y = 3; y < ch; y++)
        for (int x = 3; x < cw - 3; x++) {
            int s0 = 0, s1 = 0;
            for (int k = 0; k < n - 1; k++) {
                s0 += ccb[k] * t->cb[y + dy[k]][x + dx[k]];
                s1 += ccr[k] * t->cr[y + dy[k]][x + dx[k]];
            }
            if (g->num_y_points) {
                int luma = 0;
                const int lx = ((x - 3) << s->ssx) + 3, ly = ((y - 3) << s->ssy) + 3;
                for (int i = 0; i <= s->ssy; i++)
                    for (int j = 0; j <= s->ssx; j++)
                        luma += t->y[ly + i][lx + j];
                luma = round2(luma, s->ssx + s->ssy);
                s0 += luma * ccb[n - 1];
                s1 += luma * ccr[n - 1];
            }
            if (use_cb)
                t->cb[y][x] = (int16_t)clip3(gmin, gmax, t->cb[y][x] + round2(s0, ar_shift));
            if (use_cr)
                t->cr[y][x] = (int16_t)clip3(gmin, gmax, t->cr[y][x] + round2(s1, ar_shift));
        }
}

void av1fg_templates_batch(const Av1fgGrainShape *s, const unsigned *seeds, int n, int planes, Av1fgBatch *b,
                           Av1fgTemplates *t)
{
    enum { NB = AV1FG_BATCH };
    const Av1FilmGrain *g = &s->fg;
    const int shift = 12 - s->bit_depth + g->grain_scale_shift;
    const int center = 128 << (s->bit_depth - 8);
    const int gmin = -center, gmax = (256 << (s->bit_depth - 8)) - 1 - center;
    const int ar_shift = g->ar_coeff_shift_minus_6 + 6, ar_round = 1 << (ar_shift - 1);
    const int lag = g->ar_coeff_lag;
    const int need_luma = (planes & 1) || ((planes & 6) && g->num_y_points);
    if (n > NB)
        n = NB;
    if (need_luma) {
        if (g->num_y_points) {
            uint32_t reg[NB];
            for (int k = 0; k < NB; k++)
                reg[k] = seeds[k < n ? k : 0] & 0xffff;
            int16_t *o = &b->y[0][0][0];
            for (int i = 0; i < AV1FG_TPL_H * AV1FG_TPL_W; i++, o += NB) {
                uint32_t idx[NB];
                for (int k = 0; k < NB; k++) { /* lane-parallel LFSR step */
                    const uint32_t r = reg[k];
                    const uint32_t bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
                    reg[k] = (r >> 1) | (bit << 15);
                    idx[k] = reg[k] >> 5;
                }
                for (int k = 0; k < NB; k++)
                    o[k] = (int16_t)round2(av1fg_gaussian_sequence[idx[k]], shift);
            }
        } else {
            memset(b->y, 0, sizeof(b->y));
        }
        if (g->num_y_points) {
            int coef[24], off[24], nt = 0, i = 0;
            for (int r = -lag; r <= 0; r++)
                for (int c = -lag; c <= lag; c++) {
                    if (!r && !c)
                        break;
                    const int cf = g->ar_coeffs_y_plus_128[i++] - 128;
                    if (cf) {
                        coef[nt] = cf;
                        off[nt] = (r * AV1FG_TPL_W + c) * NB;
                        nt++;
                    }
                }
            for (int y = 3; y < AV1FG_TPL_H; y++)
                for (int x = 3; x < AV1FG_TPL_W - 3; x++) {
                    int16_t *o = b->y[y][x];
                    int32_t sum[NB] = { 0 };
                    for (int k = 0; k < nt; k++) {
                        const int16_t *q = o + off[k];
                        const int32_t c = coef[k];
                        for (int l = 0; l < NB; l++)
                            sum[l] += q[l] * c;
                    }
                    for (int l = 0; l < NB; l++) {
                        const int v = o[l] + ((sum[l] + ar_round) >> ar_shift);
                        o[l] = (int16_t)(v < gmin ? gmin : v > gmax ? gmax : v);
                    }
                }
        }
        if (planes & 1)
            for (int k = 0; k < n; k++)
                for (int y = 0; y < AV1FG_TPL_H; y++)
                    for (int x = 0; x < AV1FG_TPL_W; x++)
                        t[k].y[y][x] = b->y[y][x][k];
    }
    if (!(planes & 6) || s->mono)
        return;
    const int cw = s->ssx ? 44 : 82, ch = s->ssy ? 38 : 73;
    const int use[2] = { g->num_cb_points || g->chroma_scaling_from_luma, g->num_cr_points || g->chroma_scaling_from_luma };
    for (int pl = 0; pl < 2; pl++) {
        uint32_t reg[NB];
        for (int k = 0; k < NB; k++)
            reg[k] = (seeds[k < n ? k : 0] ^ (pl ? 0x49d8 : 0xb524)) & 0xffff;
        for (int y = 0; y < ch; y++) {
            int16_t *o = &b->c[pl][y][0][0];
            for (int x = 0; x < cw; x++, o += NB) {
                if (!use[pl]) {
                    for (int k = 0; k < NB; k++)
                        o[k] = 0;
                    continue;
                }
                uint32_t idx[NB];
                for (int k = 0; k < NB; k++) {
                    const uint32_t r = reg[k];
                    const uint32_t bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
                    reg[k] = (r >> 1) | (bit << 15);
                    idx[k] = reg[k] >> 5;
                }
                for (int k = 0; k < NB; k++)
                    o[k] = (int16_t)round2(av1fg_gaussian_sequence[idx[k]], shift);
            }
        }
    }
    int cc[2][25], off[25], nt = 0, i = 0, lc[2] = { 0, 0 };
    for (int r = -lag; r <= 0; r++)
        for (int c = -lag; c <= lag; c++) {
            if (!r && !c) {
                lc[0] = g->ar_coeffs_cb_plus_128[i] - 128;
                lc[1] = g->ar_coeffs_cr_plus_128[i] - 128;
                break;
            }
            cc[0][nt] = g->ar_coeffs_cb_plus_128[i] - 128;
            cc[1][nt] = g->ar_coeffs_cr_plus_128[i] - 128;
            off[nt] = (r * AV1FG_TPL_W + c) * NB;
            nt++;
            i++;
        }
    const int ssum = s->ssx + s->ssy, sround = ssum ? 1 << (ssum - 1) : 0;
    for (int pl = 0; pl < 2; pl++) {
        for (int y = 3; y < ch && use[pl]; y++)
            for (int x = 3; x < cw - 3; x++) {
                int16_t *o = b->c[pl][y][x];
                int32_t sum[NB] = { 0 };
                for (int k = 0; k < nt; k++) {
                    const int16_t *q = o + off[k];
                    const int32_t c = cc[pl][k];
                    for (int l = 0; l < NB; l++)
                        sum[l] += q[l] * c;
                }
                if (g->num_y_points) {
                    const int lx = ((x - 3) << s->ssx) + 3, ly = ((y - 3) << s->ssy) + 3;
                    int32_t luma[NB] = { 0 };
                    for (int a = 0; a <= s->ssy; a++)
                        for (int bb = 0; bb <= s->ssx; bb++) {
                            const int16_t *q = b->y[ly + a][lx + bb];
                            for (int l = 0; l < NB; l++)
                                luma[l] += q[l];
                        }
                    for (int l = 0; l < NB; l++)
                        sum[l] += ((luma[l] + sround) >> ssum) * lc[pl];
                }
                for (int l = 0; l < NB; l++) {
                    const int v = o[l] + ((sum[l] + ar_round) >> ar_shift);
                    o[l] = (int16_t)(v < gmin ? gmin : v > gmax ? gmax : v);
                }
            }
        for (int k = 0; k < n; k++) {
            int16_t(*c)[AV1FG_TPL_W] = pl ? t[k].cr : t[k].cb;
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                    c[y][x] = b->c[pl][y][x][k];
        }
    }
    for (int k = 0; k < n; k++) {
        t[k].cw = cw;
        t[k].ch = ch;
    }
}

void av1fg_block_offsets(unsigned seed, int width, int height, uint8_t *out, int *rows, int *cols)
{
    const int nr = ((height + 1) / 2 + 15) / 16, nc = ((width + 1) / 2 + 15) / 16;
    for (int n = 0; n < nr; n++) {
        unsigned reg = seed;
        reg ^= ((unsigned)(n * 37 + 178) & 255) << 8;
        reg ^= (unsigned)(n * 173 + 105) & 255;
        for (int b = 0; b < nc; b++)
            out[n * nc + b] = (uint8_t)get_random_number(&reg, 8);
    }
    *rows = nr;
    *cols = nc;
}
