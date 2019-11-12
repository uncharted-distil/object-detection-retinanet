import os
import sys
import warnings
import typing
import time

import keras
import keras.preprocessing.image
import tensorflow as tf
import pandas as pd
import numpy as np

import layers  
import losses
import models

from collections import OrderedDict

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame
from common_primitives import denormalize as Denormalize

from callbacks import RedirectModel
from callbacks.eval import Evaluate
from utils.eval import evaluate
from models.retinanet import retinanet_bbox
from preprocessing.csv_generator import CSVGenerator
from utils.anchors import make_shapes_callback
from utils.model import freeze as freeze_model

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    backbone = hyperparams.Union(
        OrderedDict({
            'resnet50': hyperparams.Constant[str](
                default = 'resnet50',
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = "Backbone architecture from resnet50 architecture (https://arxiv.org/abs/1512.03385)"
            ),
            'resnet101': hyperparams.Constant[str](
                default = 'resnet101',
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = "Backbone architecture from resnet101 architecture (https://arxiv.org/abs/1512.03385)"
            ),
            'resnet152': hyperparams.Constant[str](
                default = 'resnet152',
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = "Backbone architecture from resnet152 architecture (https://arxiv.org/abs/1512.03385)"
            )
        }),
        default = 'resnet50',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Backbone architecture from which RetinaNet is built. All backbones " +
                      "require a weights file downloaded for use during runtime."
    )
    batch_size = hyperparams.Hyperparameter[int](
        default = 1,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Size of the batches as input to the model."
    )
    n_epochs = hyperparams.Hyperparameter[int](
        default = 1,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of epochs to train."
    )
    freeze_backbone = hyperparams.Hyperparameter[bool](
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Freeze training of backbone layers."
    )
    weights = hyperparams.Hyperparameter[bool](
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Load the model with pretrained weights specific to selected backbone."
    )
    learning_rate = hyperparams.Hyperparameter[float](
        default = 1e-5,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Learning rate."
    )
    n_steps = hyperparams.Hyperparameter[int](
        default = 1,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of steps/epoch."
    )
    output = hyperparams.Hyperparameter[bool](
        default = False,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Output images and predicted bounding boxes after evaluation."
    )

# score-threshold [?]
# iou-threshold [?]
# max-detections [?]

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
        'name': 'retina_net',
        'python_path': 'd3m.primitives.object_detection.retinanet_convolutional_neural_network',
        'keywords': ['object detection', 'convolutional neural network', 'digital image processing', 'RetinaNet'],
        'source': {
            'name': 'Sanjeev Namjoshi',
            'contact': 'mailto:sanjeev@yonder.co',
            'uris': [
                'https://github.com/NewKnowledge/object-detection'
            ],
        },
       'installation': [
            {
                'type': 'PIP',
                'package_uri': 'git+https://github.com/NewKnowledge/object-detection.git@{git_commit}#egg=object-detection'.format(
                    git_commit = utils.current_git_commit(os.path.dirname(__file__)),)
            },
            {
            'type': "FILE",
            'key': "resnet50",
            'file_uri': "http://public.datadrivendiscovery.org/ResNet-50-model.keras.h5",
            'file_digest': "0128cdfa3963288110422e4c1a57afe76aa0d760eb706cda4353ef1432c31b9c" # TBD 
            }
        ],
        #'algorithm_types': [metadata_base.PrimitiveAlgorithmType.RETINANET_CONVOLUTIONAL_NEURAL_NETWORK],
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK],
        #'primitive_family': metadata_base.PrimitiveFamily.OBJECT_DETECTION
        'primitive_family': metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING,
        }
    )
 
    def __init__(self, *, hyperparams: Hyperparams, volumes: typing.Dict[str,str] = None) -> None:
        super().__init__(hyperparams = hyperparams, volumes = volumes)
        self.image_paths = None
        self.annotations = None
        self.base_dir = None
        self.classes = None
        self.backbone = None
        self.y_true = None
        self.workers = 1
        self.multiprocessing = 1
        self.max_queue_size = 10
        
    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ 
        Sets the primitive's training data and preprocesses the files for RetinaNet format.

        Parameters
        ----------
            inputs: numpy ndarray of size (n_images, dimension) containing the d3m Index, image name, 
                    and bounding box for each image.

        Returns
        -------
            No returns. Function is called by pipeline at runtime.
        """

        # Prepare annotation file
        ## Generate image paths
        image_cols = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        self.base_dir = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in image_cols]
        self.image_paths = np.array([[os.path.join(self.base_dir, filename) for filename in inputs.iloc[:,col]] for self.base_dir, col in zip(self.base_dir, image_cols)]).flatten()
        self.image_paths = pd.Series(self.image_paths)

        ## Prepare y_true
        #self.y_true = 

        ## Arrange proper bounding coordinates
        bounding_coords = inputs.bounding_box.str.split(',', expand = True)
        bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
        bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
        bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]

        ## Generate class names
        class_name = pd.Series(['class'] * inputs.shape[0])

        ## Assemble annotation file
        self.annotations = pd.concat([self.image_paths, bounding_coords, class_name], axis = 1)
        self.annotations.columns = ['img_file', 'x1', 'y1', 'x2', 'y2', 'class_name']

        # Prepare ID file
        self.classes = pd.DataFrame({'class_name': ['class'], 
                                     'class_id': [0]})

    def _create_callbacks(self, model, training_model, prediction_model):
        """
        Creates the callbacks to use during training.

        Parameters
        ----------
            model                : The base model.
            training_model       : The model that is used for training.
            prediction_model     : The model that should be used for validation.
            validation_generator : The generator for creating validation data.
        
        Returns
        -------
            callbacks            : A list of callbacks used for training.
        """
        callbacks = []

        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor   = 'loss',
            factor    = 0.1,
            patience  = 2,
            verbose   = 1,
            mode      = 'auto',
            min_delta = 0.0001,
            cooldown  = 0,
            min_lr    = 0
        ))

        return callbacks
    
    def _create_models(self, backbone_retinanet, num_classes, weights, freeze_backbone = False, lr = 1e-5):
        
        """ 
        Creates three models (model, training_model, prediction_model).

        Parameters
        ----------
            backbone_retinanet : A function to call to create a retinanet model with a given backbone.
            num_classes        : The number of classes to train.
            weights            : The weights to load into the model.
            multi_gpu          : The number of GPUs to use for training.
            freeze_backbone    : If True, disables learning for the backbone.
            config             : Config parameters, None indicates the default configuration.

        Returns
        -------
            model              : The base model. 
            training_model     : The training model. If multi_gpu=0, this is identical to model.
            prediction_model   : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
        """

        modifier = freeze_model if freeze_backbone else None
        anchor_params = None
        num_anchors   = None

        model = self._model_with_weights(backbone_retinanet(num_classes, num_anchors = num_anchors, modifier = modifier), weights = weights, skip_mismatch = True)
        training_model = model
        prediction_model = retinanet_bbox(model = model, anchor_params = anchor_params)
        training_model.compile(
            loss = {
                'regression'    :  losses.smooth_l1(),
                'classification':  losses.focal()
            },
            optimizer = keras.optimizers.adam(lr = lr, clipnorm = 0.001)
        )

        return model, training_model, prediction_model
    
    def _num_classes(self):
        """ 
        Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def _model_with_weights(self, model, weights, skip_mismatch):
        """ 
        Load weights for model.

        Parameters
        ----------
            model         : The model to load weights for.
            weights       : The weights to load.
            skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.

        Returns
        -------
            model         : Model with loaded weights.
        """

        if weights is not None:
            model.load_weights(weights, by_name = True, skip_mismatch = skip_mismatch)
        return model

    def _create_generator(self, annotations, classes, shuffle_groups):
        """
        Create generator for evaluation.
        """

        validation_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], self.backbone.preprocess_image, shuffle_groups = False)
        return validation_generator


    def _evaluate_model(self, generator, model, iou_threshold, score_threshold, max_detections, save_path):
        """ 
        Evaluate a given dataset using a given model.

        Parameters
        ----------
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
        
        Returns
        -------
        all_detections   : A list containing the predicted boxes for each image in the generator.
        """

        all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
        print(np.shape(all_detections), file = sys.__stdout__)

        for i in range(generator.size()):
            raw_image    = generator.load_image(i)
            image        = generator.preprocess_image(raw_image.copy())
            image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis = 0))[:3]
        #print(boxes, file = sys.__stdout__)
        #print(scores, file = sys.__stdout__)
        #print(labels, file = sys.__stdout__)

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis = 1), np.expand_dims(image_labels, axis = 1)], axis = 1)

        #print(image_boxes, file = sys.__stdout__)
        #print(image_scores, file = sys.__stdout__)
        #print(image_labels, file = sys.__stdout__)
        #print(image_detections, file = sys.__stdout__)

        # if save_path is True:
        #     draw_annotations(raw_image, generator.load_annotations(i), label_to_name = generator.label_to_name)
        #     draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name = generator.label_to_name, score_threshold = score_threshold)

        #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
        
        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        return all_detections

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Creates the image generators and then trains RetinaNet model on the image paths in the input 
        dataframe column.

        Can choose to use validation generator. 
        
        If no weight file is provided, the default is to use the ImageNet weights.
        """

        # Create object that stores backbone information
        self.backbone = models.backbone(self.hyperparams['backbone'])

        # Create the generators
        train_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], self.backbone.preprocess_image)

        # Running the model
        ## Assign weights
        if self.hyperparams['weights'] is False:
            weights = None
        else:
            weights = self.volumes[self.hyperparams['backbone']]

        ## Create model
        print('Creating model...', file = sys.__stdout__)

        model, self.training_model, prediction_model = self._create_models(
            backbone_retinanet = self.backbone.retinanet,
            num_classes = train_generator.num_classes(),
            weights = weights,
            freeze_backbone = self.hyperparams['freeze_backbone'],
            lr = self.hyperparams['learning_rate']
        )

        #print(model.summary(), file = sys.__stdout__)
        model.summary()

        ### !!! vgg AND densenet BACKBONES CURRENTLY NOT IMPLEMENTED !!!
        ## Let the generator compute the backbone layer shapes using the actual backbone model
        # if 'vgg' in self.hyperparams['backbone'] or 'densenet' in self.hyperparams['backbone']:
        #     train_generator.compute_shapes = make_shapes_callback(model)
        #     if validation_generator:
        #         validation_generator.compute_shapes = train_generator.compute_shapes
        
        ## Set up callbacks
        callbacks = self._create_callbacks(
            model,
            self.training_model,
            prediction_model,
        )

        start_time = time.time()
        print('Starting training...', file = sys.__stdout__)

        self.training_model.fit_generator(
            generator = train_generator,
            steps_per_epoch = self.hyperparams['n_steps'],
            epochs = self.hyperparams['n_epochs'],
            verbose = 1,
            callbacks = callbacks,
            workers = self.workers,
            use_multiprocessing = self.multiprocessing,
            max_queue_size = self.max_queue_size
        )
        
        print(f'Training complete. Training took {time.time()-start_time} seconds', file = sys.__stdout__)
        return CallResult(None) 


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce image detection predictions.

        Parameters
        ----------
            inputs: numpy ndarray of size (n_images, dimension) containing the d3m Index, image name, 
            and bounding box for each image.

        Returns
        -------
            outputs: 
                list of size (n_detections, dimension) where the columns are the image name followed
                by the detection coordinate for each image. The detection coordinates are in 8-coordinate
                format.

                list of size (n_images, dimensions) where the columns are the image name followed by the
                ground truth coordinate for each image. The ground truth coordinates are in 8-coordinate
                format.
        """
        iou_threshold = 0.5     # Bounding box overlap threshold for false positive or true positive
        score_threshold = 0.05  # The score confidence threshold to use for detections
        max_detections = 100    # Maxmimum number of detections to use per image
        
        # create the generator
        generator = self._create_generator(self.annotations, self.classes, shuffle_groups = False)

        # Convert training model to inference model
        inference_model = models.convert_model(self.training_model)

        # Assemble output lists
        ## Determine predicted bounding boxes (8-coordinate format, list)
        boxes = self._evaluate_model(generator, inference_model, iou_threshold, score_threshold, max_detections, self.hyperparams['output'])
        #y_pred = #

        print(boxes, file = sys.__stdout__)
        print(y_pred, file = sys.__stdout__)

        ## Compile into one tuple
        ground_truth_boxes, predicted_boxes = (y_true, y_pred)

        ## Return images if specified
        #if self.hyperparams('output') is not 'no_output':
            # Return images with boxes in specified path. See evaluate.py.
        
        return CallResult(results_df)