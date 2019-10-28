import tensorflow as tf
from d3m.metadata import base as metadata_base, hyperparams, params
import pandas as pd

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
        self.classes = None
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
        for each object detected in each image
        """

        # Prepare annotation file
        ## Generate image paths
        image_cols = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        base_dir = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in image_cols]
        self.image_paths = np.array([[os.path.join(base_path, filename) for filename in inputs.iloc[:,col]] for base_path, col in zip(base_paths, image_cols)]).flatten()

        ## Arrange proper bounding coordinates
        bounding_coords = inputs.bounding_box.str.split(',', expand = True)
        bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
        bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
        bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]

        ## Generate class names
        class_name = pd.Series(['class'] * input.shape[0])

        ## Assemble annotation file
        self.annotations = pd.concat([self.image_paths, bounding_coords, class_name], axis = 1)

        # Prepare ID file
        self.classes = pd.Series(['class,0'])

    def fit():
        # Create object that stores backbone information
        backbone = models.backbone(self.hyperparams['backbone'])

        # Create the generators
        train_generator, validation_generator = CSVGenerator(self.annotations, self.classes, self.hyperparams['batch_size'], backbone.preprocess_image)

        # Create the model
        if self.hyperparams['imagenet_weights'] is True:
            weights = # Imagenet weights
        else:
            weights = # .h5 file

        print('Creating model...')

        model, training_model, prediction_model = create_model(
            backbone_retinanet = backbone.retinanet.
            n_classes = train_generator.num_classes()
            weights = weights
            freeze_backbone = self.hyperparams('freeze_backbone')
            lr = self.hyperparams('lr')
        )

        print(model.summary())

        if self.hyperparams is False:
            validation_generator = None

        # Let the generator compute the backbone layer shapes using the actual backbone model
        if 'vgg' in self.hyperparams['backbone'] or 'densenet' in self.hyperparams['backbone']:
            train_generator.compute_shapes = make_shapes_callback(model)
            if validation_generator:
                validation_generator.compute_shapes = train_generator.compute_shapes
        
        # Callbacks
        callbacks = create_callbacks()

        # Train
        return training_model.fit_generator(
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


""" fit() checklist:
>>> [X] Backbone information
>>> >>> [] Check backbone()
>>> >>> >>> [] How to tell these functions to just load the backbone from a directory
>>> [X] Build out set_training_data()
>>> [] CSV generator
>>> >>> [] Check CSVGenerator()
>>> [] Create model
>>> >>> [] Check create_model()
>>> >>> [] Check num_classes()
>>> >>> [] Check make_shapes_callback()
>>> [] Compute backbone layer shapes
>>> [] Create callbacks
>>> [] Train
>>> >>> [] Check fit_generator()
"""

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