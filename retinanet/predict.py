"""
Script to run RetinaNet predictions and generate files for the pascalvoc tools.

The script takes the following arguments:
1. The name of the model.
2. The absolute path to the h5 file.
3. Path to testing images.
4. Name of the backbone used.

Run this file on an inference model and not a training model.

The script will then do the following:
1. Create the appropriate ground truth files for each image and put it in the groundtruths/ directory.
2. Run predictions on the inference model to create bounding box coordinates.
3. Export these bounding box coordinates to a files in predictions/ for each image.

"""

# Import
import sys
import pandas as pd
import tensorflow as tf
import glob
import numpy as np

from PIL import Image
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

# Other setup
#gpu = 0
#setup_gpu(gpu)

# Global vars
#MODEL_NAME = sys.argv[1]
#PRED_PATH = sys.argv[2]
#IMAGE_PATH = sys.argv[3]
MODEL_NAME = 'model2'
PRED_PATH = '/home/snamjoshi/Documents/models/model2_102219/'
IMAGE_PATH = '/home/snamjoshi/docker/datasets/seed_datasets_current/LL1_penn_fudan_pedestrian/LL1_penn_fudan_pedestrian_dataset/media/'

CSV_NAME = 'annotation_AWS.csv'
WD_PATH = '/home/snamjoshi/Documents/git_repos/object-detection/retinanet/'
CSV_PATH = WD_PATH + CSV_NAME
MODEL_PATH = PRED_PATH + MODEL_NAME + '.h5'

# Import files
annotations = pd.read_csv(CSV_PATH, header = None)

# Process ground truth file
## Modify bounding coordinates
annotations = annotations.iloc[:, 1:6]
annotations.columns = ['L', 'T', 'B', 'R', 'class_name']
annotations = annotations[['class_name', 'L', 'T', 'R', 'B']]

"""
>>> Add columns to file that represent the image_index
>>> For each image index row, collect and separate to a data frame
>>> Export each data frame in a separate CSV file
>>> The name of the file will follow the format: model_name + image_index + .txt
"""

## Export ground truth file
annotations.to_csv(WD_PATH + 'models/' + MODEL_NAME + '/groundtruths/' + MODEL_NAME + '_groundtruths.txt', index = False, header = False, sep = '\t')

# Run predictions on all images
#model = models.load_model(MODEL_PATH, backbone_name = sys.argv[4])
model = models.load_model(MODEL_PATH, backbone_name = 'resnet50')
labels_to_names = {0: 'pedestrian'}

## Load image list
image_list = []
for filename in glob.glob(IMAGE_PATH + '*.png'):
    image = Image.open(filename)
    image_list.append(image)

## Preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

## Process image
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

## Export object bounding boxes


#!/usr/bin/env python
# coding: utf-8



# ## Load necessary modules
# import keras
import keras

import sys
sys.path.insert(0, '../')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# Load RetinaNet model

MODEL_NAME = 'model1.h5'
PRED_PATH = '/mnt/data/predictions/'

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = PRED_PATH + MODEL_NAME

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
#labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

labels_to_names = {1: 'pedestrian'}


# ## Run detection on example

# In[ ]:


# load image
image = read_image_bgr(sys.argv[1])

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
#print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

print(boxes)
print(scores)
print(labels)

# visualize detections
#for box, score, label in zip(boxes[0], scores[0], labels[0]):
#    # scores are sorted so we can break
#    if score < 0.5:
#        break
#        
#    color = label_color(label)
#    
#    b = box.astype(int)
#    draw_box(draw, b, color=color)
#    
#    caption = "{} {:.3f}".format(labels_to_names[label], score)
#    draw_caption(draw, b, caption)
#    
#plt.figure(figsize=(15, 15))
#plt.axis('off')
#plt.imshow(draw)
#plt.show()


# In[ ]:





# In[ ]:




