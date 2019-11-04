### Running D3M locally for testing
"""
>>> [] Separate the D3M primitive from the other RetinaNet files
>>> [X] Package into a PIP installable (see: https://newknowledge.atlassian.net/wiki/spaces/data/pages/137560135/D3M+Primitive+Guide)
>>> [] Submit primitive?
>>> [] Download weights 
>>> [X] Set up Docker image with primitives and weights
>>> [] Begin testing primitive/pipeline to correct errors
>>> [] Final training on full data set (on Amazon instance)
>>> [] Code review
"""

### Issues (object_detection.py) ###
""" Hyperparams class
>>> [] Determine if formatting is correct for JSON
>>> [] Determine if choice/hyper/control is appropriate
>>> [] Expand choice parameters
>>> [] Fix weights hyperparameter
>>> [] Figure out how RetinaNet downloads weights and all that
"""

""" metadata
>>> [] Upload weights folder; email the right people to get this going
"""

""" evaluate_model() checklist:
>>> [] Determine what the hell the max detections is actually doing
>>> [] Refactor code so it just outputs detections for the input boxes
>>> [] Return coordinates of detections in 8-coordinate format as a list
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
>>> [] Fix weight loading defaults issue; make a check for the weights file
"""

""" produce() checklist:
>>> [X] Check create_generator()
>>> [X] Check load_model()
>>> [X] Check convert_model()
>>> [X] Determine how evaluation works with the D3M metrics
>>> [] Finish working on evaluate_model() checklist
>>> [] Metrics list (predictions and ground-truths) - convert to single object?
"""

### Issues (object_detection_pipeline.py)

"""

"""


""" 
>>> [] Figure out what each step of the primitive is for and what is appropriate for my use case
>>> [] Can I have two list outputs from produce() to pass to the metric evaluation?
>>> [] What is the input/output?
"""

""" WORKSPACE (object_detection.py) """
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

imagenet_weights = hyperparams.Hyperparameter[bool]{
    default = True,
    semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    description = "If true, initializes the model with pretrained imagenet weights. If false, expects an .h5 weights file."
}

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

weights = hyperparams.Hyperparameter[str]{
    default = 'image_net',
    semantic_types = ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
    description = "If 'image_net' (default), initializes the model with pretrained imagenet weights" +
                    "If 'custom', then the user is expected to reference their own weight file at runtime."
}
