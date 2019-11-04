import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

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
        'object_detection.utils.compute_overlap',
        ['/object-detection/utils/compute_overlap.pyx'],
        ),
    ]
)