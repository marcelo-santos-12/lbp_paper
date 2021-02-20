# Run:  python setup.py build_ext --inplace

import os
#import setuptools # for windows
from distutils.core import setup
from Cython.Build import cythonize

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('lbp_module/utils', parent_package, top_path)

    cythonize(['lbp_module/utils/interpolation.pyx', 'lbp_module/utils/_texture.pyx', 
               'lbp_module/utils/_texture_ilbp.pyx', 'lbp_module/utils/_texture_hlbp.pyx',
               'lbp_module/utils/_texture_elbp.pyx', 'lbp_module/utils/_texture_clbp.pyx',
               'lbp_module/utils/_histogram.pyx' ], working_path=base_path)
    config.add_extension('interpolation', sources=['lbp_module/utils/interpolation.c'])
    config.add_extension('_texture',      sources=['lbp_module/utils/_texture.c'])
    config.add_extension('_texture_ilbp', sources=['lbp_module/utils/_texture_ilbp.c'])
    config.add_extension('_texture_hlbp', sources=['lbp_module/utils/_texture_hlbp.c'])
    config.add_extension('_texture_elbp', sources=['lbp_module/utils/_texture_elbp.c'])
    config.add_extension('_texture_clbp', sources=['lbp_module/utils/_texture_clbp.c'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(configuration(top_path='').todict())
          )
