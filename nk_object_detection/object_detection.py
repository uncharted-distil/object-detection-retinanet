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


#__all__ = ('EnsembleForest',)
#logger = logging.getLogger(__name__)

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

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
        description = "If true, initializes the model with pretrained imagenet weights. If false, expects an .h5 weights file."
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

    compute_val_loss = hyperparams.Hyperparameter[bool]{
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
        description = "Compute validation loss during training."
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

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None):
        """ 
        Sets the primitive's training data and preprocesses the files for RetinaNet format.

        Parameters
        ----------
            inputs: numpy ndarray of size (n_images, dimension) containing the d3m Index, image name, 
            and bounding box for each image.
            outputs: numpy ndarray of size (n_detections, dimension) containing bounding box coordinates 
            for each object detected in each image.

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
            model: The base model.
            training_model: The model that is used for training.
            prediction_model: The model that should be used for validation.
            validation_generator: The generator for creating validation data.
        
        Returns
        -------
            A list of callbacks used for training.
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
            model            : The base model. 
            training_model   : The training model. If multi_gpu=0, this is identical to model.
            prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
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


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Creates the image generators and then trains RetinaNet model on the image paths in the input 
        dataframe column.

        Can choose to use validation generator. If no weight file is provided, the default is to use the
        ImageNet weights.
        """

        # Create object that stores backbone information
        backbone = models.backbone(self.hyperparams['backbone'])

        # Create the generators
        train_generator, validation_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], backbone.preprocess_image)

        # Create the model
        if self.hyperparams['imagenet_weights'] is True:
            weights = # Imagenet weights
        else:
            weights = # .h5 file

        print('Creating model...', file = sys.__stdout__)

        model, training_model, prediction_model = create_models(
            backbone_retinanet = backbone.retinanet.
            n_classes = train_generator.num_classes()
            weights = weights
            freeze_backbone = self.hyperparams('freeze_backbone')
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
            inputs: 

        Returns
        -------
            outputs:
        """

        # create the generator
        generator = create_generator(self.annotations, self.classes, shuffle_groups = False)

        # load the model
        print('Loading model...')
        model = models.load_model(self.training_model, backbone_name = self.hyperparams['backbone'])

        # Convert to inference model
        inference_model = models.convert_model(model)

        # Calculate mean average precision
        average_precision = evaluate(generator, model, iou_threashold, max_detections)

        #print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        #print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

        return CallResult(results_df)


""" produce() checklist:
>>> [X] Check create_generator()
>>> [X] Check load_model()
>>> [X] Check convert_model()
>>> [] Determine how evaluation works with the D3M metrics
>>> [] Print statement for the mAP
"""

""" fit() checklist:
>>> [X] Backbone information
>>> >>> [X] Check backbone()
>>> >>> >>> [X] How to tell these functions to just load the backbone from a directory?
>>> [X] Build out set_training_data()
>>> [X] CSV generator
>>> >>> [X] Check CSVGenerator()
>>> [X] Create model
>>> >>> [X] Check create_model()
>>> >>> [X] Check num_classes()
>>> >>> [X] Check make_shapes_callback()
>>> [X] Compute backbone layer shapes
>>> [X] Create callbacks
>>> [X] Train
>>> >>> [X] Check fit_generator()
"""



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




""" WORKSPACE """
    def create_generators(args, preprocess_image):
        """ 
        Create generators for training and validation.

        Parameters
        ----------
        batch_size       : Size of the batches as input to the model (hyperparameter).
        annotations      : Dataframe containing the image path, bounding coordinates, and class name for each image (created by set_training_data()).
        classes          : Series containing a class name and one-hot encodin (created by set_training_data())
        preprocess_image : Function that preprocesses an image for the network.
        """

        train_generator = CSVGenerator(annotations, classes, batch_size)

        if args.dataset_type == 'csv':
            train_generator = CSVGenerator(
                args.annotations,
                args.classes,
                transform_generator=transform_generator,
                visual_effect_generator=visual_effect_generator,
                **common_args
            )

        return train_generator, validation_generator

    weights = hyperparams.Hyperparameter[bool]{
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Initialize the model with weights from a file."
    }

test = pd.DataFrame({'class_name': ['obj1', 'obj2', 'obj3', 'obj4'], 
                     'class_id': [0, 1, 2, 3]}) 
    
test_dict = OrderedDict()
for index, row in test.iterrows():
    class_name, class_id = row
    test_dict[class_name] = class_id


self.classes = pd.Series(['class', 0], index = ['class_name', 'class_id'])

annotation_df = annotation
annotation_df.columns = ['img_file', 'x1', 'y1', 'x2', 'y2', 'class_name']
annotation_dict = OrderedDict()

for index, row in annotation_df.iterrows():
    img_file, x1, y1, x2, y2, class_name = row[:6]

    if img_file not in annotation_dict:
        annotation_dict[img_file] = []

    annotation_dict[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})


