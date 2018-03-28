import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import scipy.ndimage
from skimage.segmentation import clear_border
import pylab

import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, Convolution2D, MaxPooling3D, MaxPooling2D
from keras import layers
from keras import Model
from keras.utils import np_utils
from keras import backend as K

import keras.losses
import keras.metrics
from keras.models import load_model

'''
Simple script calculant la precision sur les fichiers TEST
'''
# change the loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def generate_list_ID(folder):
	np_list = os.listdir(folder)
	list_IDs = []
	for i,name in enumerate(np_list):
		if name[-16:-8] == "lung_img":
			list_IDs.append(name[0:-17])
	return list_IDs

def seuil_mask_pred(mask_predict):
    for i in range(0, mask_predict.shape[0]):
        for j in range(0, mask_predict.shape[1]):
            if(mask_predict[i][j] < 0.1 or i > 500 or j >500):
                mask_predict[i][j] = 0
            else:
                mask_predict[i][j] = 1
    return(mask_predict)

def homemade_precision(y_true_f,y_pred_f):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_true_f)):
        if (y_pred_f[i] == 1 and y_true_f[i] ==1):
            TP += 1
        elif (y_pred_f[i] == 0 and y_true_f[i] ==1):
            FN += 1
        elif (y_pred_f[i] == 1 and y_true_f[i] ==0):
            FN += 0
    return(TP,FP,FN)

def calcul_precision(FOLDER):
    precision_list = []
    list_IDs_subset = generate_list_ID(FOLDER)
    for i,ID in enumerate(list_IDs_subset):
        mask_real = np.load(os.path.join(FOLDER,ID +'_nodule_mask.npz.npy'))
   
        lung_to_predict = np.empty((1,512,512,1))
        lung_to_predict[0,:,:,0] = np.load(os.path.join(FOLDER,ID+'_lung_img.npz.npy'))
        predict_subset = model.predict(lung_to_predict)
        mask_predict = predict_subset[0,:,:,0]
        
        mask_predict_seuil = seuil_mask_pred(mask_predict)
        
        y_true_f = mask_real.flatten()
        y_pred_f = mask_predict_seuil.flatten()
        TP,FP,FN = homemade_precision(y_true_f,y_pred_f)
        precision = float(TP)/float(FP+FN+TP)
        
        precision_list.append(precision)
        print(i)
        print(precision)
    return precision_list



TEST_FOLDER = '/home/lucsoret/Projet/Supelec/CEI/Data/LUNA16/Images/TEST'
h5_folder = '/home/lucsoret/Projet/Supelec/CEI/Data/LUNA16/UNET/h5'

keras.losses.dice_coef_loss = dice_coef_loss
keras.metrics.dice_coef = dice_coef
model_path_init = os.path.join(h5_folder,'UNET-0')
model = load_model(model_path_init)

precision = calcul_precision(TEST_FOLDER)
print("Moyenne de la précision avec les 0")
print(np.mean(precision))
print("Moyenne de la précision sans les 0")
while 0.0 in precision:
	precision.remove(0.0)
print(np.mean(precision))