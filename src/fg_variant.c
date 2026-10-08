/*
 * Copyright © 2026, divergentnn (dav1d-fullgrain)
 * SPDX-License-Identifier: BSD-2-Clause
 * (the same terms as dav1d, see COPYING)
 *
 * Film grain synthesis variants: runtime configuration, scratch buffers and
 * statistics. See src/fg_variant.h. NON-CONFORMANT by design.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "src/cpu.h"
#include "src/fg_variant.h"
#include "src/filmgrain.h"
#include "src/mem.h"

static const char *const mode_names[] = {
    [DAV1D_FGMODE_STANDARD]   = "standard",
    [DAV1D_FGMODE_STANDARD_C] = "standard_c",
    [DAV1D_FGMODE_MULTI]      = "multi",
    [DAV1D_FGMODE_DUAL]       = "dual",
    [DAV1D_FGMODE_FULL]       = "full",
};

const char *dav1d_fgv_mode_name(const enum Dav1dFGMode mode) {
    return mode_names[mode];
}

static int env_int(const char *name, int def, int lo, int hi) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    char *end;
    const long v = strtol(s, &end, 10);
    if (*end) return def;
    return v < lo ? lo : v > hi ? hi : (int) v;
}

static int parse_mode(const char *const m) {
    for (int i = 0; i < (int) (sizeof(mode_names) / sizeof(*mode_names)); i++)
        if (!strcmp(m, mode_names[i])) return i;
    if (!strcmp(m, "std")) return DAV1D_FGMODE_STANDARD;
    if (!strcmp(m, "c")) return DAV1D_FGMODE_STANDARD_C;
    return -1;
}

/* switch mode, allocating what the new mode needs; keeps the old mode on
 * allocation failure. Only called while no grain rows are running. */
static int set_mode(Dav1dFGVariant *const fgv, const int mode) {
    if ((mode == DAV1D_FGMODE_MULTI || mode == DAV1D_FGMODE_DUAL) && !fgv->tmpl) {
        fgv->tmpl = dav1d_alloc_aligned(ALLOC_COMMON_CTX,
                                        fgv->tmpl_set_bytes * fgv->ntmpl, 64);
        if (!fgv->tmpl) return -1;
    }
    if (mode >= DAV1D_FGMODE_MULTI && fgv->simd && ARCH_X86_64 && !fgv->lut32) {
        fgv->lut32 = dav1d_alloc_aligned(ALLOC_COMMON_CTX, 3 * 4096 * sizeof(int32_t), 64);
        if (!fgv->lut32) fgv->simd = 0;
    }
    fgv->mode = mode;
    return 0;
}

COLD void dav1d_fgv_init(Dav1dFGVariant *const fgv) {
    memset(fgv, 0, sizeof(*fgv));
    fgv->mode = DAV1D_FGMODE_STANDARD;
    fgv->ntmpl = env_int("DAV1D_GRAIN_TEMPLATES", 16, 2, FGV_MAX_TMPL);
    fgv->warmup = env_int("DAV1D_GRAIN_WARMUP", 16, 4, 64) & ~1;
    fgv->bands = env_int("DAV1D_GRAIN_BANDS", 4, 1, 8);
    fgv->stats = env_int("DAV1D_GRAIN_STATS", 0, 0, 2);
    const size_t set = 3 * (GRAIN_HEIGHT + 1) * GRAIN_WIDTH * sizeof(int16_t);
    fgv->tmpl_set_bytes = (set + 63) & ~(size_t) 63;
#if ARCH_X86_64
    fgv->simd = !!(dav1d_get_cpu_flags() & DAV1D_X86_CPU_FLAG_AVX2);
#elif ARCH_AARCH64
    fgv->simd = !!(dav1d_get_cpu_flags() & DAV1D_ARM_CPU_FLAG_NEON);
#elif FGV_NEON
    fgv->simd = 1; /* WebAssembly SIMD128 build */
#endif
    if (!env_int("DAV1D_GRAIN_SIMD", 1, 0, 1)) fgv->simd = 0;
#ifdef DAV1D_GRAIN_MODE_DEFAULT
    if (parse_mode(DAV1D_GRAIN_MODE_DEFAULT) >= 0)
        set_mode(fgv, parse_mode(DAV1D_GRAIN_MODE_DEFAULT));
#endif
    const char *m = getenv("DAV1D_GRAIN_MODE");
    if (m && parse_mode(m) >= 0) set_mode(fgv, parse_mode(m));
    const char *f = getenv("DAV1D_GRAIN_MODE_FILE");
    if (f && *f) {
        const size_t n = strlen(f);
        fgv->mode_file = malloc(n + 1);
        if (fgv->mode_file) memcpy(fgv->mode_file, f, n + 1);
        dav1d_fgv_poll(fgv);
    }
    if (fgv->stats)
        fprintf(stderr, "dav1d-grain: mode=%s templates=%d warmup=%d bands=%d simd=%d%s%s\n",
                mode_names[fgv->mode], fgv->ntmpl, fgv->warmup, fgv->bands, fgv->simd,
                fgv->mode_file ? " mode_file=" : "", fgv->mode_file ? fgv->mode_file : "");
}

/* DAV1D_GRAIN_MODE_FILE: re-read the mode (first word of the file) at most
 * every 100 ms, from the per-frame grain preparation. Lets a player switch
 * modes live (see mpv/grain-mode.lua). */
void dav1d_fgv_poll(Dav1dFGVariant *const fgv) {
    if (!fgv->mode_file) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const uint64_t now = (uint64_t) ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    if (fgv->last_poll && now - fgv->last_poll < 100000000ULL) return;
    fgv->last_poll = now;
    FILE *const fp = fopen(fgv->mode_file, "r");
    if (!fp) return;
    char buf[32] = { 0 };
    const int ok = fscanf(fp, "%31s", buf) == 1;
    fclose(fp);
    if (!ok) return;
    const int mode = parse_mode(buf);
    if (mode >= 0 && mode != (int) fgv->mode) {
        set_mode(fgv, mode);
        if (fgv->stats)
            fprintf(stderr, "dav1d-grain: switched to mode=%s\n", mode_names[fgv->mode]);
    }
}

COLD void dav1d_fgv_close(Dav1dFGVariant *const fgv) {
    if (fgv->stats) {
        const uint64_t fr = atomic_load(&fgv->frames);
        const double prep = atomic_load(&fgv->ns_prep) * 1e-6;
        const double rows = atomic_load(&fgv->ns_rows) * 1e-6;
        fprintf(stderr, "dav1d-grain: mode=%s frames=%llu grain_cpu_ms=%.2f "
                "(prep %.2f + rows %.2f) per_frame_ms=%.4f (prep %.4f rows %.4f)\n",
                mode_names[fgv->mode], (unsigned long long) fr, prep + rows, prep, rows,
                fr ? (prep + rows) / fr : 0.0, fr ? prep / fr : 0.0, fr ? rows / fr : 0.0);
        if (fgv->stats > 1 && fr) {
            fprintf(stderr, "dav1d-grain: phases Mcycles/frame:");
            for (int i = 0; i < 8; i++)
                fprintf(stderr, " p%d=%.3f", i, atomic_load(&fgv->phase[i]) * 1e-6 / fr);
            fprintf(stderr, "\n");
        }
    }
    if (fgv->tmpl) dav1d_free_aligned(fgv->tmpl);
    if (fgv->lut32) dav1d_free_aligned(fgv->lut32);
    for (int i = 0; i < FGV_MAX_SLOTS; i++)
        free(fgv->slot[i]);
    free(fgv->mode_file);
    memset(fgv, 0, sizeof(*fgv));
}

void *dav1d_fgv_scratch_get(Dav1dFGVariant *const fgv, const size_t sz, int *const slot) {
    uint_fast64_t cur = atomic_load(&fgv->busy);
    for (;;) {
        if (!~cur) { *slot = -1; return malloc(sz); }
        const int i = __builtin_ctzll(~cur);
        if (atomic_compare_exchange_weak(&fgv->busy, &cur, cur | (1ULL << i))) {
            *slot = i;
            if (fgv->slot_sz[i] < sz) {
                free(fgv->slot[i]);
                fgv->slot[i] = malloc(sz);
                fgv->slot_sz[i] = fgv->slot[i] ? sz : 0;
            }
            return fgv->slot[i];
        }
    }
}

void dav1d_fgv_scratch_put(Dav1dFGVariant *const fgv, const int slot, void *const buf) {
    if (slot < 0) { free(buf); return; }
    atomic_fetch_and(&fgv->busy, ~(1ULL << slot));
}

uint64_t dav1d_fgv_thread_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
