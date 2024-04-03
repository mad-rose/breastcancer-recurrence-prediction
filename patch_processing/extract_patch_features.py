'''
Code to extract features from patches in a data directory using Img2Vec. Default CNN is set to ResNet-18.
'''

from img2vec_pytorch import Img2Vec
import os
from PIL import Image
from tqdm import tqdm
import time
import numpy as np

img2vec = Img2Vec(cuda=True)

data_dir = '.'
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")
test_dir = os.path.join(data_dir, "test")

pics = 0
target_value = 8000
data = {}
case = 0
cases = []
features = []
labels = []
with tqdm(total=target_value) as pbar:
    while pics < target_value:
        for dir_ in os.listdir(val_dir):
            print("Starting dir:", dir_)
            category = os.path.join(val_dir, dir_)
            for img_path in os.listdir(category):
                if img_path.endswith(".tiff"):
                    img_path_ = os.path.join(category, img_path)
                    img = Image.open(img_path_)
                    
                    img_features = img2vec.get_vec(img)
                    case = int(img_path[:3])
                    features.append(img_features)
                    labels.append(category)
                    cases.append(case)
                    pics = pics + 1
                    pbar.update(1)
                    time.sleep(0.1)
            print(dir_, "Directory complete")

print(data.keys())
np.save("val_labels_vgg.npy", labels, allow_pickle = True )
np.save("val_data_vgg.npy", features, allow_pickle = True )
np.save("val_cases_vgg.npy", cases, allow_pickle = True )

print("NPY arrays saved: val")


pics = 0
target_value = 22000
data = {}
case = 0
cases = []
features = []
labels = []
with tqdm(total=target_value) as pbar:
    while pics < target_value:
        for dir_ in os.listdir(test_dir):
            print("Starting dir:", dir_)
            category = os.path.join(test_dir, dir_)
            for img_path in os.listdir(category):
                if img_path.endswith(".tiff"):
                    img_path_ = os.path.join(category, img_path)
                    img = Image.open(img_path_)
                    
                    img_features = img2vec.get_vec(img)
                    case = int(img_path[:3])
                    features.append(img_features)
                    labels.append(category)
                    cases.append(case)
                    pics = pics + 1
                    pbar.update(1)
                    time.sleep(0.1)
            print(dir_, "Directory complete")

print(data.keys())
np.save("test_labels_vgg.npy", labels, allow_pickle = True )
np.save("test_data_vgg.npy", features, allow_pickle = True )
np.save("test_cases_vgg.npy", cases, allow_pickle = True )

print("NPY arrays saved: test")


pics = 0
target_value = 72000
data = {}
case = 0
cases = []
features = []
labels = []
with tqdm(total=target_value) as pbar:
    while pics < target_value:
        for dir_ in os.listdir(train_dir):
            print("Starting dir:", dir_)
            category = os.path.join(train_dir, dir_)
            for img_path in os.listdir(category):
                if img_path.endswith(".tiff"):
                    img_path_ = os.path.join(category, img_path)
                    img = Image.open(img_path_)
                    
                    img_features = img2vec.get_vec(img)
                    case = int(img_path[:3])
                    features.append(img_features)
                    labels.append(category)
                    cases.append(case)
                    pics = pics + 1
                    pbar.update(1)
                    time.sleep(0.1)
            print(dir_, "Directory complete")

print(data.keys())
np.save("train_labels_vgg.npy", labels, allow_pickle = True )
np.save("train_data_vgg.npy", features, allow_pickle = True )
np.save("train_cases_vgg.npy", cases, allow_pickle = True )


print("NPY arrays saved: train")
