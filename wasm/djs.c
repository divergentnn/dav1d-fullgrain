/*
 * dav1d-fullgrain for WebAssembly: a small C API for JavaScript.
 *
 * The API shape (open / feed data / take pictures / close) follows dav1d.js
 * by Kagami Hiiragi (https://github.com/Kagami/dav1d.js, CC0); this version
 * adds 8/10-bit output without conversion, threads, a picture queue and the
 * film-grain mode.
 *
 * Copyright © 2026, divergentnn (dav1d-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <emscripten.h>
#include <dav1d/dav1d.h>

#define QMAX 48

typedef struct djs {
    Dav1dContext *c;
    Dav1dPicture q[QMAX];
    int qh, qn;
    Dav1dPicture cur;
    int have_cur;
} djs;

/* threads: dav1d worker threads (0 = auto); delay: max frame delay (0 = auto);
 * mode: "standard", "full", "dual", "multi" or NULL/"" for the build default */
EMSCRIPTEN_KEEPALIVE
djs *djs_open(const int threads, const int delay, const char *const mode) {
    if (mode && *mode) setenv("DAV1D_GRAIN_MODE", mode, 1);
    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = threads;
    s.max_frame_delay = delay;
    s.apply_grain = 1;
    djs *const d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    if (dav1d_open(&d->c, &s) < 0) {
        free(d);
        return NULL;
    }
    return d;
}

static int pull(djs *const d) {
    while (d->qn < QMAX) {
        Dav1dPicture p;
        memset(&p, 0, sizeof(p));
        const int r = dav1d_get_picture(d->c, &p);
        if (r == DAV1D_ERR(EAGAIN)) return 0;
        if (r < 0) return r;
        d->q[(d->qh + d->qn) % QMAX] = p;
        d->qn++;
    }
    return 0;
}

/* feed one temporal unit (an MP4 sample / IVF frame); decoded pictures are
 * queued, see djs_pop. Returns 0 or a negative error. */
EMSCRIPTEN_KEEPALIVE
int djs_send(djs *const d, const uint8_t *const buf, const int sz) {
    Dav1dData data;
    memset(&data, 0, sizeof(data));
    uint8_t *const p = dav1d_data_create(&data, sz);
    if (!p) return -1;
    memcpy(p, buf, sz);
    while (data.sz) {
        const int r = dav1d_send_data(d->c, &data);
        if (r == DAV1D_ERR(EAGAIN)) {
            const int e = pull(d);
            if (e < 0 || d->qn >= QMAX) {
                dav1d_data_unref(&data);
                return e < 0 ? e : -2;
            }
            continue;
        }
        if (r < 0) {
            dav1d_data_unref(&data);
            return r;
        }
    }
    return pull(d);
}

/* after the last data: collect every remaining picture */
EMSCRIPTEN_KEEPALIVE int djs_drain(djs *const d) { return pull(d); }

EMSCRIPTEN_KEEPALIVE int djs_queued(const djs *const d) { return d->qn; }

/* make the oldest queued picture the current one; 1 if there was one */
EMSCRIPTEN_KEEPALIVE
int djs_pop(djs *const d) {
    if (d->have_cur) {
        dav1d_picture_unref(&d->cur);
        d->have_cur = 0;
    }
    if (!d->qn) return 0;
    d->cur = d->q[d->qh];
    d->qh = (d->qh + 1) % QMAX;
    d->qn--;
    d->have_cur = 1;
    return 1;
}

/* current picture */
EMSCRIPTEN_KEEPALIVE int djs_w(const djs *const d) { return d->cur.p.w; }
EMSCRIPTEN_KEEPALIVE int djs_h(const djs *const d) { return d->cur.p.h; }
EMSCRIPTEN_KEEPALIVE int djs_bpc(const djs *const d) { return d->cur.p.bpc; }
EMSCRIPTEN_KEEPALIVE int djs_layout(const djs *const d) { return d->cur.p.layout; }
EMSCRIPTEN_KEEPALIVE int djs_stride(const djs *const d, const int i) { return (int) d->cur.stride[i ? 1 : 0]; }
EMSCRIPTEN_KEEPALIVE uintptr_t djs_plane(const djs *const d, const int i) { return (uintptr_t) d->cur.data[i]; }
EMSCRIPTEN_KEEPALIVE int djs_grain(const djs *const d) {
    return d->cur.frame_hdr && d->cur.frame_hdr->film_grain.present;
}
EMSCRIPTEN_KEEPALIVE int djs_mtrx(const djs *const d) { return d->cur.seq_hdr ? d->cur.seq_hdr->mtrx : 2; }
EMSCRIPTEN_KEEPALIVE int djs_trc(const djs *const d) { return d->cur.seq_hdr ? d->cur.seq_hdr->trc : 2; }
EMSCRIPTEN_KEEPALIVE int djs_pri(const djs *const d) { return d->cur.seq_hdr ? d->cur.seq_hdr->pri : 2; }
EMSCRIPTEN_KEEPALIVE int djs_full_range(const djs *const d) { return d->cur.seq_hdr ? d->cur.seq_hdr->color_range : 0; }

/* drop queued pictures and decoder state (before decoding from a new keyframe) */
EMSCRIPTEN_KEEPALIVE
void djs_flush(djs *const d) {
    while (djs_pop(d)) {}
    dav1d_flush(d->c);
}

EMSCRIPTEN_KEEPALIVE
void djs_close(djs *const d) {
    if (!d) return;
    while (djs_pop(d)) {}
    dav1d_close(&d->c);
    free(d);
}
