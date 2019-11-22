# D3M object detection primitive using a Keras implementation of RetinaNet

The base library can be found here: https://github.com/fizyr/keras-retinanet. See section below for major changes and modifications.

## Installation

You must have `numpy` version 1.15.4 or greater to pip install this package.

Using `pip3` version 19 or greater: `pip3 install -e git+https://github.com/NewKnowledge/object-detection-retinanet#egg=object_detection_retinanet`. 

## Changes from Fizyr Keras RetinaNet implemetation

The D3M primitive is essentially a wrapper on the entire Keras-RetinaNet codebase to fit into D3M specifications. The primitive itself is found in the [object-detection-d3m-wrapper](https://github.com/NewKnowledge/object-detection-d3m-wrapper/) repo.

Since the Keras-RetinaNet codebase is a command-line tool, these details had to be stripped out and the arguments exposed as primitive hyperparameters. Most of the `train.py` script was inserted into the `fit()` and the other methods it calls were inserted into the primitive class. The only major modifications were to the `Generator` class which has to be modified slightly to parse the datasets as they are input in D3M format. 

`convert_model.py` and `evaluate.py` were inserted into the `produce()` method. `evaluate.py` has a `--convert-model` CLI argument which is mostly the same as the content in `convert_model.py`. Therefore, `convert_model.py` was removed and the contents of `evaluate.py` that convert the model were retained to be inserted into `produce()`. The modifications to `evaluate.py()` were to output a data frame that contains the list of bounding boxes in the expected format for the `metric.py` D3M evaluation using average precision.
