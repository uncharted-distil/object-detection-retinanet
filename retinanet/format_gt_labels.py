"""Test script for processing output table into metrics.py format for evaluation"""

import pandas as pd

# Global vars
#WD_PATH = '/home/snamjoshi/Documents/git_repos/object-detection/retinanet/'
WD_PATH = '~/docker/datasets/seed_datasets_current/LL1_penn_fudan_pedestrian/TRAIN/dataset_TRAIN/tables/'
#CSV_PATH = WD_PATH + 'learning_data.csv'
CSV_PATH = WD_PATH + 'learningData.csv'
IMAGES_DIR_PATH = '/home/snamjoshi/docker/datasets/seed_datasets_current/LL1_penn_fudan_pedestrian/LL1_penn_fudan_pedestrian_dataset/media/'

# Prepare and export annotations data file
learning_data = pd.read_csv(CSV_PATH)
learning_data = learning_data.drop(['d3mIndex'], axis = 1)
learning_data['image'] = IMAGES_DIR_PATH + learning_data['image']

bounding_coords = learning_data['bounding_box'].str.split(',', expand = True)
bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]

class_name = pd.Series(['pedestrian'] * learning_data.shape[0])
learning_data = learning_data.drop(['bounding_box'], axis = 1)

annotation = pd.concat([learning_data, bounding_coords, class_name], axis = 1)
annotation.columns = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']

""" Actual processing begins here"""
import os

def annotation_row_to_groundtruth_list(DataFrame):
    path = os.path.basename(DataFrame['path'])
    coords = DataFrame[['x1A', 'y1A', 'x1B', 'y2A', 'x2A', 'y2B', 'x2B', 'y1B']]
    image_box_list = [path]
    image_box_list.extend(coords)
    return image_box_list

annotation = annotation.reindex(['path', 'x1', 'y1', 'x1', 'y2', 'x2', 'y2', 'x2', 'y1'], axis = 1)
annotation.columns = ['path', 'x1A', 'y1A', 'x1B', 'y2A', 'x2A', 'y2B', 'x2B', 'y1B']
y_true = annotation.apply(annotation_row_to_groundtruth_list, axis = 1).to_list()

print(map(os.path.basename, annotation['path'].to_list())

[os.path.basename(x) for x in annotation['path'].to_list()]