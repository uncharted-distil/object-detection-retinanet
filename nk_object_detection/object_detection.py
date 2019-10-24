import tensorflow as tf
from d3m.metadata import base as metadata_base, hyperparams, params
import pandas as pd

#__all__ = ('EnsembleForest',)
#logger = logging.getLogger(__name__

Inputs = container.dataset.Dataset
Outputs = container.dataset.Dataset

class Hyperparams(hyperparams.Hyperparams):
    backbone = hyperparams.Hyperparameter[str](
        default = 'resnet50',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
        description = "Backbone architecture which RetinaNet is built. This can be one of " + 
                      "'densenet121', 'densenet169', 'densenet201'" +
                      "'mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224'" +
                      "'resnet50', 'resnet101', 'resnet152', " +
                      "'vgg16', 'vgg19" +
                      "All models require downloading weights before runtime."
    )

    batch_size = hyperparams.Hyperparameter[int](
        default = 1,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Size of the batches as input to the model."
    )

    n_epochs = hyperparams.Hyperparameter[int]{
        default = 50,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of epochs to train."
    }

    freeze_backbone = hyperparams.Hyperparameter[bool]{
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
        description = "Freeze training of backbone layers."
    }

    imagenet_weights = hyperparams.Hyperparameter[bool]{
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Initializes the model with pretrained imagenet weights."
    }

    learning_rate = hyperparams.Hyperparameter[float]{
        default = 1e-5,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Learning rate."
    }

    n_steps = hyperparams.Hyperparameter[int]{
        default = 10000,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of steps/epoch."
    }

    weights = hyperparams.Hyperparameter[bool]{
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Initialize the model with weights from a file."
    }

class Params(params.Params):
    pass

class ObjectDetectionRNPrimitive(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive that utilizes RetinaNet, a convolutional neural network (CNN), for object
    detection. The methodology comes from "Focal Loss for Dense Object Detection" by 
    Lin et al. 2017 (https://arxiv.org/abs/1708.02002). The code implementation is based
    off of the base library found at: https://github.com/fizyr/keras-retinanet.

    The primitive accepts a Dataset consisting of images, labels as input and returns
    a dataframe as output which include the bounding boxes for each object in each image.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
        'id': 'd921be1e-b158-4ab7-abb3-cb1b17f42639',
        'version': '0.1.0',
        'name': 'ObjectDetectionRN',
        'keywords': ['object detection', 'convolutional neural network'],
        'source': {
            'name': 'Sanjeev Namjoshi',
            'contact': 'sanjeev@newknowledge.io',
            'uris': [
                'https://github.com/NewKnowledge/object-detection',
            ],
        },
       'installation': [{
                'type': # TBD,
                'package_uri': # TBD,
                ),
            }],
            'algorithm_types': [
                # TBD ,
            ],
            'primitive_family': #TBD,
        },
    )
 
def __init__():
    

def __getstate__():
    

def set_training_data():


def fit():
# Preprocess input to get correct format for IDs and annotations
# Backbone information
# Create generators (augmentation happens here)
# Create model
# Compute backbone layer shapes
# Create callbacks
# Train

def produce():
# Load model
# Convert to inference model
# Create generator
# Evaluate on test data

"""Hyperparameter information"""
# Current hyperparams:
# backbone [resnet50]
# batch_size [1]
# n_epochs [50]
# freeze-backbone [True]
# imagenet_weights: initialize the model with pretrained imagenet weights [True]
# lr [float]
# n_steps [int]
# weights: initialize model with weights from a file [True]

# Default constants:
# multiprocessing: use multiprocessing in fit_generator [True]
# workers: number of generator workers [1]
# max-queue-size: queue length for multiprocessing workers in fit_generator [10]

# Possible future hyperparams:
# gpu: Id of GPU to use
# multi-gpu: number of GPUs to use
# model-type: regression or classification. Is this necessary??

# Image augmentation options:
# randomly transform image and annotations
# image-min-side: rescale the image so the smallest side is min_side
# image-max-side: rescale the image if the largest side is larger than max_side

