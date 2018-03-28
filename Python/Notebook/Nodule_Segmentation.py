#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:45:11 2018

@author: anthonypamart
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label,regionprops
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import distance


##Fonction pour faire des plot en 3D
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces, x, y = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    

##Applique le nodule_mask au lung_img et renvoie donc le nodule_segmented
def get_nodule_segmented(nodule_mask,lung_img):
    nodule_segmented = lung_img.copy()
    for i in range(0,len(nodule_segmented)):
        for j in range(0,len(nodule_segmented)):
            if(nodule_mask[i][j]==0.0):
                nodule_segmented[i][j]=0.0
    return nodule_segmented

##Prend le nodule mask et renvoie les coordonnées des centres des nodules présents et le nb de nodules présents
def get_centers_nodules(nodule_mask):
    label_scan = label(nodule_mask)
    center_full = [r.centroid  for r in regionprops(label_scan)]
    nb_nodules = len(center_full)
    return center_full, nb_nodules

##Pour récupérer le path des lung_img pour le patient dont on précise l'ID
##Attention, si l'INPUT FOLDER change, il faut faire varier les indices dans la condition if pour bien toujours de récupérer l'ID du patient à partir de images_path_lung_img[i]
def get_patients_path_lung_img(images_path_lung_img, ID='1.3.6.1.4.1.14519.5.2.1.6279.6001.137763212752154081977261297097'):
    patients_path_lung_img = []
    for i in range(0,len(images_path_lung_img)):
        #if(images_path_lung_img[i][67:131]==id_patients[0]):
        if(images_path_lung_img[i][67:131]==ID):
            patients_path_lung_img.append(images_path_lung_img[i])
    patients_path_lung_img.sort()
    return patients_path_lung_img

##Pour récupérer le path des nodule_mask pour le patient dont on précise l'ID
def get_patients_path_nodule_mask(images_path_nodule_mask, ID='1.3.6.1.4.1.14519.5.2.1.6279.6001.137763212752154081977261297097'):
    patients_path_nodule_mask = []
    for i in range(0,len(images_path_nodule_mask)):
        if(images_path_nodule_mask[i][67:131]==ID):
            patients_path_nodule_mask.append(images_path_nodule_mask[i])
    patients_path_nodule_mask.sort()
    return patients_path_nodule_mask


##On met en entrée le path des nodule_mask du patient et on a en sortie un np array 3D avec le nodule mask (2D)
##de toutes les slices (ce qui ajoute une 3e dimension)
def get_nodule_mask_3D(patients_path_nodule_mask, dim_x=512, dim_y=512):
    nodule_mask_3D = np.zeros((dim_x, dim_y, len(patients_path_nodule_mask)))
    for i in range(0,len(patients_path_nodule_mask)):
        temp = np.load(patients_path_nodule_mask[i]).reshape(512,512)
        
        for k in range(0, temp.shape[0]):
            for j in range(0, temp.shape[1]):
                if(temp[k][j] < 0.1 or i > 500 or j >500):
                    temp[k][j] = 0
                else:
                    temp[k][j] = 1
        
        nodule_mask_3D[:,:,i] = temp
    
    return nodule_mask_3D

##On met en entrée le path des lung_img du patient et on a en sortie un np array 3D avec la lung_img (2D)
##de toutes les slices (ce qui ajoute une 3e dimension)
def get_lung_img_3D(patients_path_lung_img, dim_x=512, dim_y=512):
    lung_img_3D = np.zeros((dim_x, dim_y, len(patients_path_lung_img)))
    for i in range(0,len(patients_path_lung_img)):
        lung_img_3D[:,:,i] = np.load(patients_path_lung_img[i]) ##Pas ouf de loader à chaque fois ??
    return lung_img_3D

##On met en entrée le path des lung_img du patient et le path des nodule_mask et on a en sortie un np array 3D
##avec le nodule segmenté (2D) de toutes les slices (ce qui rajoute une 3e dimension)
def get_nodule_segmented_3D(patients_path_nodule_mask, patients_path_lung_img, dim_z = 32, dim_x =512, dim_y=512):
    nodule_segmented_3D = np.zeros((dim_x, dim_y, dim_z))
    
    a= min(len(patients_path_lung_img), dim_z)
    for i in range(0,a):
        nodule_segmented_3D[:,:,i] = get_nodule_segmented(get_nodule_mask_3D(patients_path_nodule_mask)[:,:,i],get_lung_img_3D(patients_path_lung_img)[:,:,i])
    return nodule_segmented_3D

##On stock les center_full et les nb_nodules dans des listes center_full_3D et nb_nodules_3D
def get_center_nodules_3D(patients_path_nodule_mask):
    center_full_3D = []
    nb_nodules_3D = []

    for i in range(0, len(patients_path_nodule_mask)):
        nodule_mask_3D = get_nodule_mask_3D(patients_path_nodule_mask)
        center_full, nb_nodules = get_centers_nodules(nodule_mask_3D[:,:,i])
        center_full_3D.append(center_full)
        nb_nodules_3D.append(nb_nodules)  
    return center_full_3D, nb_nodules_3D


##A partir des paths vers lesquels on trouve lung_img et mask_nodule et de l'ID du patient, on récupère un npdule 3D de taille 32x32x32
def npz_to_cnn(images_path_lung_img, images_path_nodule_mask, ID, dimx_nodule = 32, dimy_nodule = 32, dimz_nodule = 32):
    patients_path_lung_img = get_patients_path_lung_img(images_path_lung_img, ID)
    patients_path_nodule_mask = get_patients_path_nodule_mask(images_path_nodule_mask, ID)
    nodule_segmented_3D = get_nodule_segmented_3D(patients_path_nodule_mask, patients_path_lung_img, dimz_nodule)
    center_full_3D, nb_nodules_3D = get_center_nodules_3D(patients_path_nodule_mask)
    a = min(len(center_full_3D), dimz_nodule)
    
    first_nodule_3D = np.zeros((dimx_nodule, dimy_nodule, dimz_nodule)) 
    
    #for i in range(0,len(nb_nodules_3D)):
    for i in range(0,a):
        first_nodule_3D[:,:,i] = nodule_segmented_3D[int(center_full_3D[i][-1][0])-dimx_nodule/2:int(center_full_3D[i][-1][0])+dimx_nodule/2,int(center_full_3D[i][-1][1])-dimy_nodule/2:int(center_full_3D[i][-1][1])+dimy_nodule/2,i]
    return first_nodule_3D, center_full_3D
