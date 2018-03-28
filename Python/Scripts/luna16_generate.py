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
import dicom
import scipy.misc

import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import scipy.ndimage
from skimage.segmentation import clear_border


"""Ce script permet à partir des images du LUNA16 dataset, generer les tranches contenant des nodules (information contenue dans le csv)"""

#Charge l i mage .mhd
def load_itk(filename):
    # Lire l image grace a SimpleITL
    itkimage = sitk.ReadImage(filename)
    
    # Convertir l image en numpy array
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Origine du ct scan, utilise plus tard pour convertir en coordonnees reelles 
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Donne l espace selon chaque dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing

#Passe de coordonnee reelles en voxel
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
Passe de voxel en coordonees reelles en utilisant l'origine et l'espacement du ct scan
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])

'''
Cree une sphere dans le masque binaire autour de la position donnee, avec le rayon donne
cands est issu du fichier annotations.csv
'''

def draw_circles(image,cands,origin,spacing):

    RESIZE_SPACING = [1, 1, 1]
    image_mask = np.zeros(image.shape)

    for ca in cands.values:
        #gOn obtient les coordonnes du nodule
        radius = np.ceil(ca[4])/2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_z,coord_y,coord_x))

        #Determine les coordonnees voxel a partir des coordonnes reelle
        image_coord = world_2_voxel(image_coord,origin,spacing)

        #Determine la range du nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        #Creation du masque
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
                    if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:
                        
                        mask_x = int(max(0,min(image_mask.shape[0]-1,coords[0])))
                        mask_y = int(max(0,min(image_mask.shape[1]-1,coords[1])))
                        mask_z = int(max(0,min(image_mask.shape[2]-1,coords[2])))

                        image_mask[mask_x,mask_y,mask_z] = int(1)

    return image_mask

def get_segmented_lungs(im):
    
    '''
    Segmente le poumon pour une slice 2D, cf le notebook pre_processing
    '''
    

    
    #Step 1: Convertion en image binaire, seuil de 604
    
    binary = im < 604
    
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


def segment_lung_from_ct_scan(ct_scan):
	return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

'''
Entree :
imagepath : chemin vers le .mhd
cands : liste des nodules associés
imageName : nom a donner à l image
path : chemin ou est enregistre l i mage

Sortie : image du poumon avec un zero-padding, et masque associe

Cree le masque a l aide de la fonction draw-circle et des informations sur les nodules contenues dans cands
'''
def create_nodule_mask(imagePath, cands, imageName,path):
    img, origin, spacing = load_itk(imagePath)

    #Meme problematique de re-sampling que dans le Kaggle dataset : on cherche a obtenir un ecart unitaire entre chaque voxel pour chaque scanner
    RESIZE_SPACING = [1, 1, 1]
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize
    
    #Resize de l image (meme demarche que le ressampling du pre_processing.py)
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
    
    #Segmente le poumon
    lung_img = lung_img + 1024
    lung_mask = segment_lung_from_ct_scan(lung_img)
    lung_img = lung_img - 1024

    #Cree le masque du nodule
    nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)

    lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros((lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))

    original_shape = lung_img.shape	
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = np.round(offset/2)
        lower_offset = offset - upper_offset

        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

        lung_img_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
        lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]
        nodule_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]

    # Sauvegarde les images, mais non utilise ici   
    #np.save(path + imageName + '_lung_img.npz', lung_img_512)
    #np.save(path + imageName + '_lung_mask.npz', lung_mask_512)
    #np.save(path + imageName + '_nodule_mask.npz', nodule_mask_512)
    
    return lung_img_512,nodule_mask_512


if __name__ == "__main__":

	'''
	Prend un subset en INPUT, un folder de stockage, et la liste des candidates
	A partir de l image 3D du scanner, sauvegarde le masque de toutes les tranches possedant un nodule
	'''
    
	INPUT_FOLDER = '/home/lucsoret/Projet/Supelec/CEI/Data/LUNA16/Images/subset4'
	cands = pd.read_csv('/home/lucsoret/Projet/Supelec/CEI/Data/LUNA16/csv/annotations.csv')

	processed_folder = '/home/lucsoret/Projet/Supelec/CEI/Data/LUNA16/Images/processed/'
	slice_folder = '/home/lucsoret/Projet/Supelec/CEI/Data/LUNA16/Images/processed/subset4/slices'

	patients_short = os.listdir(INPUT_FOLDER)

	images = os.listdir(INPUT_FOLDER)
	images_path = [os.path.join(INPUT_FOLDER,patient) for patient in patients_short]

	##On ne garde que les images en .mhd
	images_mhd = []

	for i in range(0,len(images)):
		if images[i][-4:] == '.mhd':
			images_mhd.append(images[i])

	images_mhd_path = [os.path.join(INPUT_FOLDER,mhd) for mhd in images_mhd]



   	##Liste des cands pour chaque image, on depasse un peu mais rien de grave
	cands_list = []
	for i in range(0,len(images_mhd)):
		cands_list.append(cands[cands['seriesuid'] == images_mhd[i][0:-4]])




	for pos,mhd_path in enumerate(images_mhd_path):
	
		#Inutile de gener un masque pour un patient qui n'a pas de nodules
		if (len(cands[cands['seriesuid'] ==  images_mhd[pos][0:-4]]) > 0):
			im_mhd = images_mhd[pos]
			lung,nodule = create_nodule_mask(mhd_path, cands_list[pos], im_mhd[0:-4],processed_folder)
			print(pos)
			if (np.sum(nodule) != 0):
				for s in range(lung.shape[0]):
					lung_slice = lung[s]
					nodule_slice = nodule[s]
					if (np.sum(nodule_slice) != 0):
						print('pos : {} , slice : {}'.format(str(pos),str(s)))
						print(im_mhd[0:-4])
						np.save(slice_folder + '/' + im_mhd[0:-4] + "_" + str(s) + '_lung_img.npz' , lung_slice)
						np.save(slice_folder + '/' + im_mhd[0:-4] + "_" + str(s) + '_nodule_mask.npz'  , nodule_slice)                                
	
