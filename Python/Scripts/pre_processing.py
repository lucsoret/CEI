import os
import dicom
import pylab
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy


from scipy import ndimage as ndi
from skimage.filters import roberts, sobel
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing





def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
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

def get_segmented_lungs(im,seuil=-500):
    
    '''
    Segmente le poumon pour une slice 2D, cf le notebook pre_processing
    '''
    

    
    #Step 1: Convertion en image binaire, seuil de 604
    
    binary = im < seuil
    
    #Step 2: Enleve les bords de l'image pour garder que le contenu
    
    cleared = clear_border(binary)
  
    #Step 3: On labelise l image, cette operation permet de scinder une image binaire en differente zone, selon les contacts entre les pixels

    label_image = label(cleared)

    
    #Une fois la labelisation faite, on va garder les deux regions les plus grandes, correspondants ainsi aux deux poumons
    
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    #Step 4: Erosion pour separer les contacts avec les vaisseaux sanguins
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    
    #Step 5: Closure, pour garder les nodules attachees aux parois des poumons
    selem = disk(10)
    binary = binary_closing(binary, selem)
    
    #Step 6: Remplissage
    
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    
    #Step 7: Application du masque a l image d entree
    
    get_high_vals = binary == 0
    im[get_high_vals] = 0

    return im

def normalize(image):
	MIN_BOUND = -1024
	MAX_BOUND = 400
	image = image.astype(np.float16)
	image = image / (MAX_BOUND - MIN_BOUND)
	return image




def center(image):
    #Trop complique de calculer sur toutes les donnees la moyenne, 0.25 est la moyenne empirique, ca sera suffisant
    #POUR L INSTANT, ON NE SE SERT PAS DU CENTERING (cf notebook), ON REVIENDRA PEUT ETRE ICI PLUS TARD
    im_mean = 0
    image = image-im_mean
    return image

def resample(input_image, scan, new_spacing=[1,1,1]):
    
    #L espacement tel qu il est donne
    prior_spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    #Le facteur de dilatation pour avoir notre nouvel espacement
    resize_factor = prior_spacing / new_spacing
    #Nouvel dimension de l image totale (cad toutes les slices)
    new_real_shape = input_image.shape * resize_factor
    #Mieux avec quelque chose d'entier
    new_shape = np.round(new_real_shape)
    
    real_resize_factor = new_shape / input_image.shape
    new_spacing = prior_spacing / real_resize_factor    
    #La fonction qui gere tout : scipy.ndimage
    output_image = scipy.ndimage.interpolation.zoom(input_image, real_resize_factor, mode='nearest')
    return output_image, new_spacing


def zero_padding(im,max_z=400,max_x=450,max_y=450):
    #Les dimensions sont prises empiriquement, au vu des shape montrees precedemment 
    borders = (max_z,max_x,max_y)
    fill_total = [[0,0],[0,0],[0,0]]
    for i in range(3):
        current_shape = im.shape[i]
        if (current_shape < borders[i]):
            fill = borders[i] - current_shape
            fill_left = fill/2
            fill_right = fill/2 + fill % 2
            fill_total[i] = (fill_left,fill_right)
    im = np.pad(im,((fill_total[0][0],fill_total[0][1]),(fill_total[1][0],fill_total[1][1]),(fill_total[2][0],fill_total[2][1])),'constant',constant_values=0)
    #Si jamais notre image depasse les dimensions du zero padding, on va la cut de base
    im = im[0:max_z,0:max_x,0:max_y]
    return im

#Fait tout le pre-processing sur une image
def full_process(patient_path,max_z=400,max_x=450,max_y=450):
	patient = load_scan(patient_path)
	image = PixelData_to_HU(patient)
	image,space = resample(image,patient)
	image = np.stack([get_segmented_lungs(s) for s in image])
	image =  np.stack([center(normalize(s)) for s in image])
	image = zero_padding(image,max_z,max_x,max_y)
	return image

