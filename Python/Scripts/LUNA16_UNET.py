import numpy as np
import os

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, Convolution2D, MaxPooling3D, MaxPooling2D
from keras import layers
from keras import Model
from keras.optimizers import *
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras import models
from keras import optimizers
from keras.callbacks import TensorBoard
import keras.losses
import keras.metrics
from keras.models import load_model

import numpy as np

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, dim_x = 512, dim_y = 512, dim_z = 1, batch_size = 32, shuffle = True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def generate(self,list_IDs,INPUT_FOLDER):
        #'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                # Generate data
                X, Y = self.__data_generation(list_IDs_temp,INPUT_FOLDER)

                yield X, Y
                    
    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self,list_IDs_temp,INPUT_FOLDER):
        #'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        Y = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        # Generate data

        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            X[i, :, :, 0] = np.load(INPUT_FOLDER + '/' + ID + '_lung_img.npz.npy')
            # Store class
            Y[i,:,:,0] = np.load(INPUT_FOLDER + '/' + ID + '_nodule_mask.npz.npy')
        return X, Y

# change the loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def generate_unique_logpath(logdir, raw_run_name):
        i = 0
        while(True):
                run_name = raw_run_name + "-" + str(i)
                log_path = os.path.join(logdir, run_name)
                if not os.path.isdir(log_path):
                        return log_path
                i = i + 1


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def unet():
	inputs = Input((512,512,1))

	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

	model = Model(input = inputs, output = conv10)

	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

	model.summary()

	return model

def generate_list_ID(folder):
	np_list = os.listdir(folder)
	list_IDs = []
	for i,name in enumerate(np_list):
		if name[-16:-8] == "lung_img":
			list_IDs.append(name[0:-17])
	return list_IDs


if __name__ == "__main__":

	TRAIN_FOLDER = '/usr/users/promo2017/soret_luc/Projets/CEI/Data/LUNA16/Images/processed/TRAIN'
	VAL_FOLDER = '/usr/users/promo2017/soret_luc/Projets/CEI/Data/LUNA16/Images/processed/VAL'


	list_IDs_TRAIN = generate_list_ID(TRAIN_FOLDER)
	list_IDs_VAL = generate_list_ID(VAL_FOLDER)

	#Tensorboard
	run_name = "UNET"
	logpath = generate_unique_logpath("/usr/users/promo2017/soret_luc/Projets/CEI/Data/LUNA16/UNET/logs_linear", run_name)
	tbcb = TensorBoard(log_dir=logpath)

	# Parameters
	params = {'dim_x': 512,
	          'dim_y': 512,
	          'dim_z': 1,
	          'batch_size': 4,
	          'shuffle': True}

	training_generator = DataGenerator(**params).generate(list_IDs_TRAIN,TRAIN_FOLDER)
	validation_generator = DataGenerator(**params).generate(list_IDs_VAL,VAL_FOLDER)

	#A changer si from scratch
	keras.losses.dice_coef_loss = dice_coef_loss
	keras.metrics.dice_coef = dice_coef
	model_path_init = '/usr/users/promo2017/soret_luc/Projets/CEI/Data/LUNA16/UNET/h5/UNET-0'
	model = load_model(model_path_init)
	#model = unet()

	history = model.fit_generator(generator = training_generator,
	                    steps_per_epoch = 20,
	                    epochs = 10,
	                    validation_data = validation_generator,
	                    validation_steps = 20,
	                    callbacks=[tbcb])
	model_path = generate_unique_logpath('/home/lucsoret/Projet/Supelec/CEI/Data/LUNA16/UNET/h5',run_name)
	model.save(model_path)