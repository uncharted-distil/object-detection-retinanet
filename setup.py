import setuptools
import numpy

from setuptools import setup
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

setup(
    name              = 'object-detection-retinanet',
    version           = '0.1.0',
    description       = 'Keras implementation of RetinaNet',
    url               = 'https://github.com/NewKnowledge/object-detection-retinanet/',
    author            = 'Sanjeev Namjoshi',
    author_email      = 'sanjeev@yonder.co',
    packages          = ['object_detection_retinanet'],
    install_requires  = ['keras',
                         'keras-resnet==0.1.0',
                         'six',
                         'scipy',
                         'cython>=0.28',
                         'Pillow',
                         'progressbar2',
                         'opencv-python',
                         'numpy>=1.14.0'
                        ],
    ext_modules       = [
        Extension('utils.compute_overlap', ['utils/compute_overlap.pyx'],
        include_dirs = [numpy.get_include()])
    ]
)