# av1-grain-seed: replace unlucky AV1 film grain seeds in existing files
# (header parse and a 16-bit in-place edit per frame; no decode, no deps).
#
# Copyright © 2026, divergentnn (av1-fullgrain)
# SPDX-License-Identifier: BSD-2-Clause

av1fg_lib = static_library('av1fg',
    files(
        'av1hdr.c',
        'av1fg_container.c',
        'av1fg_scan.c',
        'av1fg_seeds.c',
        'av1fg_score.c',
        'av1fg_grain.c',
        'av1fg_gauss.c',
    ),
    dependencies : [thread_dependency, libm_dependency],
    c_args : cc.get_supported_arguments(['-fno-fast-math', '-ffp-contract=off']),
    override_options : ['c_std=c11'],
    install : false,
)

av1_grain_seed = executable('av1-grain-seed',
    files('av1-grain-seed.c'),
    link_with : av1fg_lib,
    dependencies : [thread_dependency, libm_dependency],
    override_options : ['c_std=c11'],
    install : true,
)

subdir('tests')
