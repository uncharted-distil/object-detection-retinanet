import setuptools

from setuptools import setup
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)

setup(
    name              = 'object-detection',
    version           = '0.1.0',
    description       = 'Keras implementation of RetinaNet as a D3M primitive',
    url               = 'https://github.com/NewKnowledge/object-detection',
    author            = 'Sanjeev Namjoshi',
    author_email      = 'sanjeev@yonder.co',
    packages          = ['object_detection'],
    install_requires  = ['keras',
                         'six',
                         'scipy',
                         'cython>=0.28',
                         'Pillow',
                         'opencv-python',
                         'numpy>=1.14.0'
                        ],
    entry_points      = {
        'd3m.primitives': [
            'object_detection.retinanet_convolutional_neural_network = object_detection:ObjectDetectionRNPrimitive'
        ],
    },
    ext_modules       = [
        Extension(
        'utils.compute_overlap',
        ['utils/compute_overlap.pyx'],
        ),
    ]
)