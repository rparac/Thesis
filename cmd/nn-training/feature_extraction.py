"""
Extracts frames from a video, which are usually used.
This script is designed to be run on the DoC machines
"""

import os

import imageio
import numpy as np
from natsort import natsorted

FEATURE_EXTRACTOR = "ResNet50V2"
# FEATURE_EXTRACTOR = "ResNet101V2"
# FEATURE_EXTRACTOR = "InceptionV3"

large_data_dir = "/vol/bitbucket/rp218/Thesis_Data"

if (FEATURE_EXTRACTOR == "ResNet50V2"):
    from keras.applications import ResNet50V2
    from keras.applications.resnet_v2 import preprocess_input
    model = ResNet50V2(weights='imagenet', include_top = False)
    features_dir = f'{large_data_dir}/feature_extraction_test/Feature_vectors_{FEATURE_EXTRACTOR}'

elif (FEATURE_EXTRACTOR == "ResNet101V2"):
    from keras.applications import ResNet101V2
    from keras.applications.resnet_v2 import preprocess_input
    model = ResNet101V2(weights='imagenet', include_top = False)
    features_dir = f'{large_data_dir}/feature_extraction_test/Feature_vectors_{FEATURE_EXTRACTOR}'


elif (FEATURE_EXTRACTOR == "InceptionV3"):
    from keras.applications import InceptionV3
    from keras.applications.inception_v3 import preprocess_input
    model = InceptionV3(weights='imagenet', include_top = False)
    features_dir = f'{large_data_dir}/feature_extraction_test/Feature_vectors_{FEATURE_EXTRACTOR}'

else:
    raise "Feature extractor selected is not supported"

video_dir = f'{large_data_dir}/Frames'

os.makedirs(features_dir, exist_ok=True)

for filename in os.listdir(video_dir):
    vid_path = os.path.join(video_dir, filename)
    frames = []

    try:
    
        for framename in natsorted(os.listdir(vid_path)):
            frame_path = os.path.join(vid_path, framename)
            frames.append(imageio.imread(frame_path))
        X = np.stack(frames, axis=0)
        X = preprocess_input(X)
        features = model.predict(X)
        features = np.max(features, axis=(1,2))
        feature_path =  os.path.join(features_dir, filename.split('.')[0])
        np.save(feature_path,features.T)

    except:
        print(f'Missing frames {filename}')