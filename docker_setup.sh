#!/bin/bash

# Install packages
pip3 install -e git+https://gitlab.com/datadrivendiscovery/common-primitives@dataset_sample#egg=common_primitives
pip3 install -e /object-detection/

# Copy pipeline file, JSON, and metadata to home
cp /object-detection/object_detection/object_detection_pipeline.py ~
cp /object-detection/787bb5eb-7ba0-4f34-8af3-0b277337e4b4.* ~

# Copy weights files to home and convert name to sha256 sum
#wget https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-50-model.keras.h5
cp /object-detection/*.h5 ~

cd ~
mv ResNet-50-model.keras.h5 0128cdfa3963288110422e4c1a57afe76aa0d760eb706cda4353ef1432c31b9c
mv model2.h5 20e0fa7483c40af0e4b5bf0d3e3f1847e2469065f7d4dbc99f451565cf9e8c90
mv model3.h5 a523326ab232565f5aed00f882c1f276b4749cae497a8f4fa4ef0049a83792ec

echo 'object-detection debugging setup complete.'