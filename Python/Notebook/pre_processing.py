import os
import dicom
import pylab
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.filters import roberts, sobel
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing





def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    return slices



def PixelData_to_HU(slices):
    #Stack de nos slices dans une tenseur 3D
    image = np.stack([s.pixel_array for s in slices])


    # Les pixels "aberrant (- 2000) sont associes a l'air, qui vaut -1000HU, le rescale valant en general -1024 on fait :
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 24
    
    # Convertion en HU
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        image[slice_number] = slope * image[slice_number] + intercept
            
    
    return np.array(image)



def get_segmented_lungs(im):
    
    #Seuillage
    binary = im < -500
    
    #Gestion du bord
    binary = clear_border(binary)
    
    #Closing
    selem = disk(10)
    binary = binary_closing(binary, selem)
    
    #Filling
    edges = roberts(binary)
    binary= ndi.binary_fill_holes(edges)
    
    #Masque
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    
    return im

def normalize(image):
	MIN_BOUND = -1024
	MAX_BOUND = 400
	image = image.astype(np.float16)
	image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
	image[image>1] = 1.
	image[image<0] = 0.
	return image




def center(image):
    #Trop complique de calculer sur toutes les donnees la moyenne, 0.25 est la moyenne empirique, ca sera suffisant
    im_mean = 0.25
    image = image-im_mean
    return image