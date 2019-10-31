# D3M object detection primitive using a Keras implementation of RetinaNet

The base library can be found here: https://github.com/fizyr/keras-retinanet

## Installation

Using `pip3` version 19 or greater: `pip3 install -e git+https://github.com/NewKnowledge/object-detection.git#egg=nk_object_detection`

## Input

File containing the D3M file index, the image name, and a string of bounding coordinates in the following format: `x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min`.

## Output

A tuple containing two lists:

1. The bounding box predictions for each image. Format: `[image_name, x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min, confidence_score].`

2. Ground truth bounding boxes for each image. Format: `[image_name, x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]`.

## Pipeline

The `object_detection_pipeline.py` file is to be run in d3m runtime with a weights file. During testing, mean average precision (mAP) is used to score the prediction results.