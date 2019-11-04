from setuptools import setup

setup(
    name              = 'nk-object-detection',
    version           = '0.1.0',
    description       = 'Keras implementation of RetinaNet as a D3M primitive',
    url               = 'https://github.com/NewKnowledge/object-detection',
    author            = 'Sanjeev Namjoshi',
    author_email      = 'sanjeev@yonder.co',
    packages          = ['nk-object-detection'],
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
)