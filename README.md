# D3M object detection primitive using a Keras implementation of RetinaNet

The base library can be found here: https://github.com/fizyr/keras-retinanet. See section below for major changes and modifications.

## Installation

Using `pip3` version 19 or greater: `pip3 install -e git+https://github.com/NewKnowledge/object-detection#egg=object-detection`

## Input

File containing the D3M file index, the image name, and a string of bounding coordinates in the following format: `x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min`.

## Output

A tuple containing two lists:

1. The bounding box predictions for each image. Format: `[image_name, x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min, confidence_score].`

2. Ground truth bounding boxes for each image. Format: `[image_name, x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]`.

## Pipeline

The `object_detection_pipeline.py` file is to be run in d3m runtime with a weights file. During testing, mean average precision (mAP) is used to score the prediction results.

## Changes from Fizyr Keras RetinaNet implenetation

The D3M primitive is essentially a wrapper on the entire Keras-RetinaNet codebase to fit into D3M specifications. Since the Keras-RetinaNet codebase is a command-line tool, these details had to be stripped out and the arguments exposed as primitive hyperparameters. Most of the `train.py` script was inserted into the `fit()` and the other methods it calls were inserted into the primitive class. The only major modifications were to the `Generator` class which has to be modified slightly to parse the datasets as they are input in D3M format.

`convert_model.py` and `evaluate.py` were inserted into the `produce()` method. `evaluate.py` has a `--convert-model` CLI argument which is mostly the same as the content in `convert_model.py`. Therefore, `convert_model.py` was removed and the contents of `evaluate.py` that convert the model were retained to be inserted into `produce()`. The modifications to `evaluate.py()` were to allow the option to output the images and also output a single tuple that contains the list of bounding boxes in the expected format for the `metric.py` evaluation using mean average precision.
