import numpy as np
import os
from pre_processing import full_process

'''Un exemple de changement du constructeur DataGenerator pour l adapter a ntore cas'''
class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 450, dim_y = 450, dim_z = 400, batch_size = 1, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle
#list_IDs = INPUT_FOLDER
  def generate(self, labels, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [k for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(labels, list_IDs_temp,list_IDs)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      patients = os.listdir(list_IDs)
      indexes = np.arange(len(patients))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, labels, list_IDs_temp,list_IDs):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      # Add 1 at the end, because de depth of the chanel is 1
      X = np.empty((self.batch_size, self.dim_z, self.dim_x, self.dim_y,1))
      y = np.empty((self.batch_size), dtype = int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # ID is i in patient(i)
          patients = os.listdir(list_IDs)
          patient_path = os.path.join(list_IDs,patients[ID])
          im = full_process(patient_path,self.dim_z,self.dim_x,self.dim_y)
          X[i, :, :, :,0] = im

          # Store class
          y[i] = labels[i]

      return X, y

