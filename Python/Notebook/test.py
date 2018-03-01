import numpy as np
import keras
from keras.models import Sequential
from my_classes import DataGenerator
from keras import optimizers
from keras import layers
from keras import models
from keras import Sequential
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TEST_FOLDER = '/home/lucsoret/Projet/Supelec/CEI/Data/sample_images'
VAL_FOLDER = '/home/lucsoret/Projet/Supelec/CEI/Data/sample_validation_images'


# Parameters
params = {'dim_x': 10,
          'dim_y': 10,
          'dim_z': 10,
          'batch_size': 1,
          'shuffle': True}

train_folder = TEST_FOLDER
val_folder = VAL_FOLDER

#S OCCUPER DE LA PARTIE LABEL QUAND ON VA VRAIMENT L UTILISER
labels = np.array([1,0,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0])


# Generators
training_generator = DataGenerator(**params).generate(labels, train_folder)
validation_generator = DataGenerator(**params).generate(labels, val_folder)




model = models.Sequential()
model.add(layers.Conv3D(32, (3, 3,3), activation='relu',
input_shape=(params['dim_z'], params['dim_x'], params['dim_y'],1)))
model.add(layers.MaxPooling3D((2, 2, 2)))
#model.add(layers.Conv3D(64, (3, 3,3), activation='relu'))
#model.add(layers.MaxPooling3D((2, 2,2)))
#model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
#model.add(layers.MaxPooling3D((2, 2, 2)))
#model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
#model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

model.fit_generator(generator = training_generator,
                    steps_per_epoch = 20,
                    epochs = 10,
                    validation_data = validation_generator,
                    validation_steps = 20)