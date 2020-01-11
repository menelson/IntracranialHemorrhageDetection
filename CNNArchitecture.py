'''
   A basic CNN implementation using keras, with a TensorFlow backend, 
   including maximum pooling, additional dense layers, and 
   Dropout for regularization. ReLu activation is used, with SoftMax also
   applied at the final dense layer.
'''
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from os import listdir
from os.path import isfile, join
import pydicom
import cv2
import time
import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

# Retrieve the training and sample submission data
training_df = pd.read_csv('./input/rsna-intracranial-hemorrhage-detection/stage_2_train.csv') # Need the labels (mapped to the ID of each pixel)
submission_df = pd.read_csv('./input/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv')

# Some feature engineering
training_df['filename'] = training_df['ID'].apply(lambda x: "ID_" + x.split('_')[1] + ".png")
training_df['type'] = training_df['ID'].apply(lambda x: x.split('_')[2])
submission_df['filename'] = submission_df['ID'].apply(lambda x: "ID_" + x.split('_')[1] + ".png")
submission_df['type'] = submission_df['ID'].apply(lambda x: x.split('_')[2])

print('Will now print some properties of the input data: ')
print('Class imbalance: ', training_df.Label.value_counts())
print('Header (20 entires): ', training_df.head(100))
print('Shape of training data: ', training_df.shape)

# Begin to look at the samples files, which will need processing 
np.random.seed(2020)
sample_files = np.random.choice(os.listdir('./input/rsna-intracranial-hemorrhage-detection/stage_2_sample_images/'), 400000)
sample_df = training_df[training_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]
pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label').reset_index()

save_and_resize(filenames=sample_files, load_dir='./input/rsna-intracranial-hemorrhage-detection/stage_2_sample_images/')

'''
  Helper functions defined below
'''

# Windowing and resizing of the images
def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

# Resize the images to 256*256
def save_and_resize(filenames, load_dir):    
    save_dir = '/kaggle/tmp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in tqdm(filenames):
        path = load_dir + filename
        new_path = save_dir + filename.replace('.dcm', '.png')
        
        dcm = pydicom.dcmread(path)
        window_center , window_width, intercept, slope = get_windowing(dcm)
        img = dcm.pixel_array
        img = window_image(img, window_center, window_width, intercept, slope)
        
        resized = cv2.resize(img, (224, 224))
        res = cv2.imwrite(new_path, resized)

# Imagine preprocessing
def preprocess_image(image, sigmaX=10):
    '''
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (size, size))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image

# CNN architecture
def CNN():
    model = models.Sequential()
    model.add(Dropout(0.2)) # Add Dropout for regularization
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.2)) 
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # For multiclass classification
   
    model.summary() # Display the architecture 