import setuptools
import numpy

from setuptools import setup, find_packages
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

setup(
    name              = 'object-detection-retinanet',
    version           = '0.1.1',
    description       = 'Keras implementation of RetinaNet',
    url               = 'https://github.com/uncharted-distil/object-detection-retinanet/',
    author            = 'Sanjeev Namjoshi',
    author_email      = 'sanjeev@yonder.co',
    license           = 'Apache-2.0',
    #packages          = ['object_detection_retinanet'],
    packages          = find_packages(),
    include_package_data = True,
    install_requires  = ['Keras==2.3.1',
                         'keras-resnet==0.1.0',
                         'six',
                         'scipy==1.4.1',
                         'cython>=0.29.24',
                         'Pillow==7.1.2',
                         'progressbar2',
                         'opencv-python',
                         'numpy>=1.15.4,<=1.18.2'
                        ],
    ext_modules       = [
        Extension('object_detection_retinanet.utils.compute_overlap', ['object_detection_retinanet/utils/compute_overlap.pyx'],
        include_dirs = [numpy.get_include()])
    ]
)
