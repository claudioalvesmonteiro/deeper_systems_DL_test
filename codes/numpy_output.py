#======================================#
# deeper systems neural networks test
# @ claudio
#=====================================#

# import packages 
import pandas as pd
import numpy as np
import os
import cv2
import pickle 

#============================#
# load and preprocess data 
#===========================#

# directory and files
dire = 'results/rotated/'
list_images = os.listdir(dire)

# make data
data = []
for i in range(len(list_images)):
    img_array = cv2.imread(os.path.join(dire,list_images[i]))
    data.append([img_array])

# save as pickle
def pickleSave(data, filename):
    picout = open(('results/'+filename), 'wb')
    pickle.dump(data, picout)
    picout.close()

# save
pickleSave(data,'numpy_output.pickle')
