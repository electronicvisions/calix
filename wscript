import os
from os.path import join
from waflib.extras.test_base import summary
from waflib.extras.symwaf2ic import get_toplevel_path


def depends(dep):
    dep("code-format")
    dep("haldls")
    dep("libnux")


def options(opt):
    opt.load('pytest')
    opt.load('pylint')
    opt.load('pycodestyle')


def configure(cfg):
    cfg.load('python')
    cfg.check_python_version()
    cfg.load('pytest')
    cfg.load('pylint')
    cfg.load('pycodestyle')


def build(bld):
    bld.env.DLSvx_HARDWARE_AVAILABLE = "cube" == os.environ.get("SLURM_JOB_PARTITION")

    bld(name='calix_pylib',
        features='py pylint pycodestyle',
        source=bld.path.ant_glob('src/py/**/*.py'),
        use='dlens_vx',
        install_path='${PREFIX}/lib',
        install_from='src/py',
        relative_trick=True,
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle")
        )

    bld(name='calix_pyswtests',
        tests=bld.path.ant_glob('tests/sw/py/**/*.py'),
        features='pytest pylint pycodestyle',
        use='calix_pylib',
        install_path='${PREFIX}/bin/tests',
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle")
        )

    bld(name='calix_pyhwtests',
        tests=bld.path.ant_glob('tests/hw/py/**/*.py'),
        features='pytest pylint pycodestyle',
        use='calix_pylib',
        install_path='${PREFIX}/bin/tests',
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        skip_run=not bld.env.DLSvx_HARDWARE_AVAILABLE
        )

    bld.program(features='cxx',
                target='template_HelloWorld.bin',
                source=['src/ppu/calix/HelloWorld.cpp'],
                use=['nux_vx', 'nux_runtime_vx'],
                env=bld.all_envs['nux_vx'],
                )

    bld.add_post_fun(summary)
