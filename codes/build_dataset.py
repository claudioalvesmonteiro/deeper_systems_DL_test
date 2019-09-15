#======================================#
# deeper systems neural networks test
# @ claudio
#=====================================#

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle 
import tensorflow as tf 

#==========================#
# PREPROCESSING
#=========================#

#==================== LOAD AND SAVE TRAIN AND TEST DATA

# load label
lb = pd.read_csv('data/train.truth.csv')

# directory and files
tdir = 'data/train/'

# label to index
indlab = list(lb['label'].unique())
dict_labels = {}
for label in indlab:
    dict_labels[label] = indlab.index(label)

# make data
data = []
for i in range(len(lb)):
    img_array = cv2.imread(os.path.join(tdir,lb['fn'][i]), cv2.IMREAD_GRAYSCALE)
    img_array = tf.keras.utils.normalize(img_array, axis=1)
    img_array = np.array(img_array).reshape(-1, 64*64)
    label = lb['label'][i]
    label_ind = dict_labels[label]
    data.append([img_array, label_ind])

# split in train and test
random.shuffle(data)
train = data[:(round(len(data)*0.8))]
test = data[(round(len(data)*0.8)):]

# divide features and labels
def generateFeatureLabel(data):
    features = []
    labels = []
    for x, y in data:
        features.append(x)
        labels.append(y)
    return np.array(features), np.array(labels)

# generate train and test 
train_feature, train_label = generateFeatureLabel(train)
test_feature, test_label = generateFeatureLabel(test)

# save as pickle
def pickleSave(data, filename):
    picout = open(('data/generated/'+filename), 'wb')
    pickle.dump(data, picout)
    picout.close()

pickleSave(train_feature,'train_feature.pickle')
pickleSave(train_label,'train_label.pickle')
pickleSave(test_feature,'test_feature.pickle')
pickleSave(test_label,'test_label.pickle')

#==================== LOAD AND SAVE EVALUATION

# directory and files
evaldir = 'data/test/'
list_eval_images = os.listdir(evaldir)

# make data
eval_data = []
for i in range(len(list_eval_images)):
    img_array = cv2.imread(os.path.join(evaldir,list_eval_images[i]), cv2.IMREAD_GRAYSCALE)
    img_array = tf.keras.utils.normalize(img_array, axis=1)
    img_array = np.array(img_array).reshape(-1, 64*64)
    eval_data.append([img_array])

# save as pickle
pickleSave(eval_data,'eval_data.pickle')