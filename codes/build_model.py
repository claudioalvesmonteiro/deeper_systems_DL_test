#======================================#
# deeper systems neural networks test
# @ claudio
#=====================================#

# import packages
import tensorflow as tf
import pickle
import numpy as np
import os
import pandas as pd

#============================#
# load and preprocess data
#===========================#

# pickle import
def pickleImport(filename):
    pickle_in = open(('data/generated/'+filename), 'rb')
    data = pickle.load(pickle_in)
    return data

# import data
feature_train = pickleImport('train_feature.pickle')
label_train = np.array(pickleImport('train_label.pickle'))

feature_test = pickleImport('test_feature.pickle')
label_test = np.array(pickleImport('test_label.pickle'))

#========================#
# BUILD MODEL
#========================#

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))# or sigmoid
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

model.compile(optimizer ='adam', # or gradient descent
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#==========================#
# train and test model
#=========================#

# train
model.fit(feature_train, label_train, epochs=15)

# test
val_loss, val_acc = model.evaluate(feature_test, label_test)

# train2
model.fit(feature_test, label_test, epochs=10)

#==========================#
# predict evaluation data
#========================#

# import final test data
eval_data = pickleImport('eval_data.pickle')

# predict
predictions = model.predict([eval_data])
predictions_value = [np.argmax(x) for x in predictions]

#======= make test.preds.csv

# archives
evaldir = 'data/test/'
list_eval_images = os.listdir(evaldir)

# predictions_keys
predict = {'rotated_left': 0, 'upright': 1, 'rotated_right': 2, 'upside_down': 3}
predictions_keys = [list(predict.keys())[list(predict.values()).index(x)] for x in predictions_value]

# dataframe
eval_preds = pd.DataFrame(predictions_keys, list_eval_images).reset_index()
eval_preds.columns = ['fn','label']

# save
eval_preds.to_csv('results/test.preds.csv', index=False)
