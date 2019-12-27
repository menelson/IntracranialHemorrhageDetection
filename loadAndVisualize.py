'''
   This code will work with CSV file as inputs, so use
   Pandas for the initial loading of data, and matplotlib
   and Seaborn for the visualization.
'''

# Imports for the data visualization and data-frame construction
import glob, pylab, pandas as pd
import pydicom, numpy as np # We need pydicom to acces the DICOM format of the medical imaging files
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
import os
import seaborn as sns


trainingData = pd.read_csv('./input/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')

print 'Will now print some properties of the input data: '
print 'Header (20 entires): ', trainingData.head(20)
print 'Shape of training data: ', trainingData.shape()

trainImagesDir = './input/rsna-intracranial-hemorrhage-detection/stage_2_train_images/'
trainImages = [f for f in listdir(trainImagesDir) if isfile(join(trainImagesDir, f))]
testImagesDir = './input/rsna-intracranial-hemorrhage-detection/stage_2_test_images/'
testImages = [f for f in listdir(testImagesDir) if isfile(join(testImagesDir, f))]
print '5 Training images', train_images[:5] # Print the first 5 images in the training set

print 'Number of training images: ', len(trainImages)
print 'Number of testing images: ', len(testImages)

# Loop over and look at the various DICOM files
fig = plt.figure(figsize=(15, 10))
# Just pick out a few images, defined by the columns*rows
columns = 5
rows = 4
for i in range(1, columns*rows + 1):
    ds = pydicom.dcmread(trainImagesDir + trainImages[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot

# Let's also add some additional new types to the table
trainingData['Sub_type'] = trainingData['ID'].str.split("_", n = 3, expand = True)[2]
trainingData['PatientID'] = trainingData['ID'].str.split("_", n = 3, expand = True)[1]

# Let's do some more manipulation: sum and display by subgroup type
groupSub = trainingData.groupby('Sub_type').sum()

# Seaborn provides nice plotting features
sns.barplot(y=groupSub.index, x=groupSub.Label, palette="deep")

fig = plt.figure(figsize=(10, 8))
sns.countplot(x="Sub_type", hue="Label", data=trainingData)
plt.title("Total Images by Subtype")
