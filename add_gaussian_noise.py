# -*- coding: utf-8 -*-
"""
Introducing Gaussian noise to data

@author: Pia Görts
"""

#import the needed package
import os
import matplotlib.pyplot as plt 
import SimpleITK as sitk

#%%
#Function to perform gaussian blurring with a given variance

def gaussian_blur(args):
    """
    Parameters
    ----------
    args : input arguments given by user, first argument is the fullpath
    of the input file, second argument is the variance (sigma) of the desired gaussian blur 

    Returns
    -------
    blur_array : the Gaussian blurred image
    
    Packages needed
    -------
    import sys
    import SimpleITK as sitk

    """

    reader = sitk.ImageFileReader()
    reader.SetFileName(args[0])
    input_image = reader.Execute()

    pixelID = input_image.GetPixelID()

    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(float(args[1]))
    blur_image = gaussian.Execute(input_image)

    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    blur_image = caster.Execute(blur_image)
    
    # Convert SimpleITK images to NumPy arrays
    blur_array = sitk.GetArrayFromImage(blur_image)
    
    return blur_array


#%%
#use the function gaussian blur on the prostate images, plot first the original image
# and then 3 times a blurred image with increasing variance

FOLDER_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Elastix'
DATA_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Data\TrainingData'
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
TRANSFORMIX_PATH = os.path.join(FOLDER_PATH, 'transformix.exe')


subfolders = os.listdir(DATA_PATH)[1:]
filenames = ["mr_bffe", "mr_bffe.zraw","prostaat","prostaat.zraw"]

for i in range(len(subfolders)):
    patient = subfolders[i]
    sigma = [1,2]
    currentpath_mr = os.path.join(DATA_PATH, patient, 'mr_bffe.mhd')
    currentpath_mask = os.path.join(DATA_PATH,patient,'prostaat.mhd')
    image_mr = sitk.ReadImage(currentpath_mr)
    array_mr = sitk.GetArrayFromImage(image_mr)
    image_mask = sitk.ReadImage(currentpath_mask)
    
    depth = 50
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].imshow(array_mr[depth,:,:], cmap='gray')
    ax[0].set_title('mr image not blurred')
    
    #Blur the image for 3 different variances of Gaussian blur
    blur_arrays = []
    for j in range(len(sigma)):
        args = [currentpath_mr,sigma[j]]
        blur_array = gaussian_blur(args)
        blur_arrays.append(blur_array)
        ax[j+1].imshow(blur_array[depth,:,:], cmap='gray')
        ax[j+1].set_title(f'mr image blurred, variance {sigma[j]}')
        
        # Save the blurred images
        OUTPUT_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Data'
        output_folder = os.path.join(OUTPUT_PATH, f'blurred_level_{j+1}')
        os.makedirs(output_folder, exist_ok=True)
        output_subfolder = os.path.join(output_folder,patient)
        os.makedirs(os.path.join(output_folder,patient),exist_ok=True)
        output_path_mr = os.path.join(output_subfolder, 'mr_bffe.mhd')
        output_path_mask = os.path.join(output_subfolder,'prostaat.mhd')
        sitk.WriteImage(sitk.GetImageFromArray(blur_array), output_path_mr)
        sitk.WriteImage(image_mask,output_path_mask)
        
        

        
        




    



