# av1-grain-seed tests: containers, edits, determinism, and (suite av1fg-dav1d)
# the parser and the edits checked against libdav1d.
#
#   meson test -C build --suite av1fg --suite av1fg-dav1d

av1fg_dav1d_check = executable('av1fg-dav1d-check',
    files('av1fg_dav1d_check.c'),
    link_with : [libdav1d, dav1d_input_objs],
    include_directories : [dav1d_inc_dirs, include_directories('../..')],
    dependencies : [thread_dependency],
    build_by_default : false,
    install : false,
)

av1fg_py = import('python').find_installation(required : false)
if av1fg_py.found()
    av1fg_test_args = [
        files('run_tests.py'),
        av1_grain_seed,
        join_paths(meson.current_source_dir(), 'fixtures'),
        '--extra', join_paths(meson.current_source_dir(), '..', '..', '..', 'tests', 'fullgrain'),
    ]
    test('av1-grain-seed', av1fg_py,
        args : av1fg_test_args,
        suite : 'av1fg',
        timeout : 600,
    )
    test('av1-grain-seed vs libdav1d', av1fg_py,
        args : av1fg_test_args + ['--dav1d-check', av1fg_dav1d_check],
        suite : 'av1fg-dav1d',
        timeout : 600,
    )
endif
