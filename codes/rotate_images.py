#======================================#
# deeper systems neural networks test
# @ claudio
#=====================================#

# import packages
import pandas as pd
from PIL import Image

# import data with predictions
predata =  pd.read_csv('results/test.preds.csv')

# function to define rotation
def rotation(image, prediction):
    if prediction == 'upside_down':
        transposed = image.transpose(Image.ROTATE_180)
    elif prediction == 'rotated_right':
        transposed  = image.transpose(Image.ROTATE_90)
    elif prediction == 'rotated_left':
        transposed  = image.transpose(Image.ROTATE_270)
    else:
        return image
    return transposed

# loop to rotate
for i in range(len(predata)):
    image  = Image.open(('data/test/' + predata['fn'][i]))
    rotated = rotation(image, predata['label'][i])
    rotated.save('results/rotated/'+predata['fn'][i])
