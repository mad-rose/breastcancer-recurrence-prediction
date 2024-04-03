'''
The code in this file can be used to aggregate patch features into bags. This code combines all cases into one large dataset. If desired, the code can be adjusted to keep the preset train/test/validation splits.
'''

import numpy as np
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


datadir = "256_2k/"


#Load extracted features
val_features = np.load(datadir2000 + "val_data_256_2k.npy", allow_pickle = True)
test_features = np.load(datadir2000 + "test_data_256_2k.npy", allow_pickle = True)
train_features = np.load(datadir2000 + "train_data_256_2k.npy", allow_pickle = True)
val_labels = np.load(datadir2000 + "val_labels_256_2k.npy", allow_pickle = True)
test_labels = np.load(datadir2000 + "test_labels_256_2k.npy", allow_pickle = True)
train_labels = np.load(datadir2000 + "train_labels_256_2k.npy", allow_pickle = True)
val_cases = np.load(datadir2000 + "val_cases_256_2k.npy", allow_pickle = True)
test_cases = np.load(datadir2000 + "test_cases_256_2k.npy", allow_pickle = True)
train_cases = np.load(datadir2000 + "train_cases_256_2k.npy", allow_pickle = True)

#Standardize labels
val_labels2 = [x[11:] for x in val_labels]
val_labels = val_labels2
test_labels2 = [x[7:] for x in test_labels]
test_labels = test_labels2
train_labels2 = [x[8:] for x in train_labels]
train_labels = train_labels2

vttcases = []
for x in val_cases:
    vttcases.append(x)
for x in train_cases:
    vttcases.append(x)
for x in test_cases:
    vttcases.append(x)
vttcases = set(vttcases)
print(vttcases)
print(len(vttcases))

i = 0
vtt_bag_features = []
vtt_bag_labels = []
vtt_bag_cases = []
for x in vttcases:
    bag = []
    bag_labels = []
    for y in range(len(val_cases)):
        if x == val_cases[y]:
            bag.append(val_features[y])
            label = val_labels[y]
    for y in range(len(test_cases)):
        if x == test_cases[y]:
            bag.append(test_features[y])
            label = test_labels[y]
    for y in range(len(train_cases)):
        if x == train_cases[y]:
            bag.append(train_features[y])
            label = train_labels[y]
    vtt_bag_labels.append(label)
    vtt_bag_features.append(bag)
    vtt_bag_cases.append(x)

np.save("256_2k_features.npy", vtt_bag_features, allow_pickle = True)
np.save("256_2k_labels.npy", vtt_bag_labels, allow_pickle = True)
np.save("256_2k_cases.npy", vtt_bag_cases, allow_pickle = True)