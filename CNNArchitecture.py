'''
   A basic CNN implementation using keras, with a TensorFlow backend, 
   including maximum pooling, additional dense layers, and 
   Dropout for regularization. ReLu activation is used, with SoftMax also
   applied at the final dense layer.
'''

# Imports for the modelling
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def CNN():
    model = models.Sequential()
    model.add(Dropout(0.2)) # Add Dropout for regularization
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.2)) # Add Dropout for regularization
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2)) # Add Dropout for regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
   
    model.summary() # Display the architecture 
