import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf
import pandas as pd

from d3m.metadata import base as metadata_base, hyperparams, params
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..utils.eval import evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.model import freeze as freeze_model
#from ..utils.gpu import setup_gpu

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    backbone = hyperparams.Choice(
        choices = {
            'densenet121': hyperparams.Hyperparams[None](default = None),
            'densenet169': hyperparams.Hyperparams[None](default = None),
            'densenet201': hyperparams.Hyperparams[None](default = None),
            'mobilenet128': hyperparams.Hyperparams[None](default = None),
            'mobilenet160': hyperparams.Hyperparams[None](default = None),
            'mobilenet192': hyperparams.Hyperparams[None](default = None),
            'mobilenet224': hyperparams.Hyperparams[None](default = None),
            'resnet50': hyperparams.Hyperparams[None](default = None),
            'resnet101': hyperparams.Hyperparams[None](default = None),
            'resnet152': hyperparams.Hyperparams[None](default = None),
            'vgg16': hyperparams.Hyperparams[None](default = None),
            'vgg19': hyperparams.Hyperparams[None](default = None)
        },
        default = 'resnet50',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
        description = "Backbone architecture from which RetinaNet is built. All models " +
                      "requires a weights file downloaded for use during runtime."
    )

    batch_size = hyperparams.Hyperparameter[int](
        default = 1,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Size of the batches as input to the model."
    )

    n_epochs = hyperparams.Hyperparameter[int](
        default = 50,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of epochs to train."
    )

    freeze_backbone = hyperparams.Hyperparameter[bool](
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControleParameter'],
        description = "Freeze training of backbone layers."
    )

    weights = hyperparams.Choice(
        choices = {
            'imagenet': hyperparams.Hyperparams[None](default = None),
            'custom': hyperparams.Hyperparams[None](default = None)
        },
        default = 'image_net',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
        description = "If 'imagenet' (default), initializes the model with pretrained imagenet weights" +
                     "If 'custom', then the user is expected to reference their own weight file at runtime."
    )

    learning_rate = hyperparams.Hyperparameter[float](
        default = 1e-5,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Learning rate."
    )

    n_steps = hyperparams.Hyperparameter[int](
        default = 10000,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of steps/epoch."
    )

    compute_val_loss = hyperparams.Hyperparameter[bool](
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Compute validation loss during training."
    )


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
        'keywords': ['object detection', 'convolutional neural network', 'digital image processing', 'RetinaNet'],
        'source': {
            'name': 'Sanjeev Namjoshi',
            'contact': 'sanjeev@newknowledge.io',
            'uris': [
                'https://github.com/NewKnowledge/object-detection',
            ],
        },
       'installation': [
            {
                'type': "PIP",
                'package_uri': "" # TBD
            },
            {
                "type": "FILE",
                "key": "weights",
                "file_uri": "",   # TBD
                "file_digest": "" # TBD 
            }
        ],
        'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.RETINANET_CONVOLUTIONAL_NEURAL_NETWORK 
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OBJECT_DETECTION
        }
    )
 
    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams = hyperparam)
        self.image_paths = None
        self.annotations = None
        self.base_dir = None
        self.classes = None
        self.training_model = None
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
        self.image_paths = np.array([[os.path.join(base_dir, filename) for filename in inputs.iloc[:,col]] for base_dir, col in zip(base_dir, image_cols)]).flatten()

        ## Arrange proper bounding coordinates
        bounding_coords = inputs.bounding_box.str.split(',', expand = True)
        bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
        bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
        bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]

        ## Generate class names
        class_name = pd.Series(['class'] * input.shape[0])

        ## Assemble annotation file
        self.annotations = pd.concat([self.image_paths, bounding_coords, class_name], axis = 1)
        self.annotation.columns = ['img_file', 'x1', 'y1', 'x2', 'y2', 'class_name']

        # Prepare ID file
        self.classes = pd.DataFrame({'class_name': ['class'], 
                                     'class_id': [0]})


    def create_callbacks(model, training_model, prediction_model, validation_generator):
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

    
    def create_models(backbone_retinanet, num_classes, weights, multi_gpu = 0, 
                      freeze_backbone = False, lr = 1e-5, config = None):
                      
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

        model = model_with_weights(backbone_retinanet(num_classes, num_anchors = num_anchors, modifier = modifier), weights = weights, skip_mismatch = True)
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
    
    def num_classes(self):
        """ 
        Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1


    def create_generator(args):
        """
        Create generator for evaluation.
        """

        validation_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], backbone.preprocess_image, shuffle_groups = False)
        return validation_generator


    def evaluate_model(generator, model, iou_threshold, score_threshold, max_detections):
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
        all_dtections   : A list containing the predicted boxes for each image in the generator.
        """

        for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

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
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

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
        backbone = models.backbone(self.hyperparams['backbone'])

        # Create the generators
        train_generator, validation_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], backbone.preprocess_image)

        # Create the model
        # Check for weights if 'custom_weights' were uploaded6
        if self.hyperparams['weights'] == 'imagenet':
            weights = imagenet_weights
        else:
            weights = custom_weights

        print('Creating model...', file = sys.__stdout__)

        model, training_model, prediction_model = create_models(
            backbone_retinanet = backbone.retinanet,
            n_classes = train_generator.num_classes(),
            weights = weights,
            freeze_backbone = self.hyperparams('freeze_backbone'),
            lr = self.hyperparams('lr')
        )

        print(model.summary(), file = sys.__stdout__)

        if self.hyperparams is False:
            validation_generator = None

        # Let the generator compute the backbone layer shapes using the actual backbone model
        if 'vgg' in self.hyperparams['backbone'] or 'densenet' in self.hyperparams['backbone']:
            train_generator.compute_shapes = make_shapes_callback(model)
            if validation_generator:
                validation_generator.compute_shapes = train_generator.compute_shapes
        
        # Callbacks
        callbacks = create_callbacks()

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
            max_queue_size = self.max_queue_size,
            validation_data = validation_generator
        )
        
        print(f'Training complete. training took {time.time()-start_time} seconds', file = sys.__stdout__)
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
        generator = create_generator(self.annotations, self.classes, shuffle_groups = False)

        # load the model
        print('Loading model...')
        model = models.load_model(self.training_model, backbone_name = self.hyperparams['backbone'])

        # Convert to inference model
        inference_model = models.convert_model(model)

        # Assemble output lists
        ## Determine predicted bounding boxes (8-coordinate format, list)
        boxes = evaluate_model(generator, model, iou_threshold, score_threshold, max_detections)

        ## Convert ground truth bounding boxes to list (ground_truth_list)

        ## Compile into one object??
        
        return CallResult(results_df)
