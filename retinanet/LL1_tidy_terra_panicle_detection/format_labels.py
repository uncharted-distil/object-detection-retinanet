"""
Script to format learning_data.csv into a format accepted by RetinaNet.
RetinaNet expects the annotations to be of the form 'path/to/image.png,x1,y1,x2,y2,class_name'

This script is modified to work on the LL1_tidy_terra_panicle_detection dataset.
"""

# Import
import pandas as pd

# Global vars
WD_PATH = '/home/snamjoshi/Documents/git_repos/object-detection/retinanet/LL1_tidy_terra_panicle_detection/'
CSV_PATH = WD_PATH + 'learningData.csv'
IMAGES_DIR_PATH = '/home/snamjoshi/docker/datasets/seed_datasets_current/LL1_tidy_terra_panicle_detection/LL1_tidy_terra_panicle_detection/media/'
IMAGES_DIR_PATH_AWS = '/mnt/data/datasets/datasets/seed_datasets_current/LL1_tidy_terra_panicle_detection/LL1_tidy_terra_panicle_detection_dataset/media/'

# Prepare and export annotations data file
learning_data = pd.read_csv(CSV_PATH)
learning_data = learning_data.drop(['d3mIndex'], axis = 1)
#learning_data['image'] = IMAGES_DIR_PATH + learning_data['image']
learning_data['image'] = IMAGES_DIR_PATH_AWS + learning_data['image']

bounding_coords = learning_data['bounding_box'].str.split(',', expand = True)
bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]

class_name = pd.Series(['panicle'] * learning_data.shape[0])
learning_data = learning_data.drop(['bounding_box'], axis = 1)

annotation = pd.concat([learning_data, bounding_coords, class_name], axis = 1)
annotation.columns = ['path', 'x1', 'y1', 'x2', 'y2', 'class_name']
#annotation.to_csv(WD_PATH + 'annotation.csv', index = False, header = False)
annotation.to_csv(WD_PATH + 'annotation_AWS.csv', index = False, header = False)

# Prepare and export ID data file
id_mapping = pd.concat([annotation['class_name'], pd.Series(annotation.index.values)], axis = 1)
id_mapping.columns = ['class_name', 'id']
id_mapping = id_mapping.iloc[[1]]

id_mapping.to_csv(WD_PATH + 'id_mapping.csv', index = False, header = False)
