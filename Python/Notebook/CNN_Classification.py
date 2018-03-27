#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:17:23 2018

@author: anthonypamart
"""

from keras import layers
from keras import models
from keras import optimizers


def CNN_Classification():
    image_channels = 1
    
    model = models.Sequential()
    
    model.add(layers.Conv3D(32, (3,3,3), activation = 'relu', input_shape = (32,32,32,image_channels)))
    model.add(layers.MaxPooling3D((2,2,2)))
    
    model.add(layers.Conv3D(64, (3,3,3), activation = 'relu'))
    model.add(layers.MaxPooling3D((2,2,2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
              
    model.add(layers.Dense(216, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(108, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(1, kernel_initializer='normal', activation='sigmoid')) ##Couche de sortie

    #model.summary()
        
    model.compile(loss='binary_crossentropy',
                          optimizer=optimizers.Adam(lr=1e-3))
    

    
    return model

    
