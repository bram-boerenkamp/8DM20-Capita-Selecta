#%%
import elastix
import imageio
import os
import matplotlib.pyplot as plt
import numpy
import SimpleITK as sitk 

FOLDER_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA'
DATA_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA\TrainingData'
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
TRANSFORMIX_PATH = os.path.join(FOLDER_PATH, 'transformix.exe')

#%%


patients = ['p102', 'p107']

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for i, item in enumerate(patients):
    i = i*2
    image_mr = sitk.ReadImage(os.path.join(DATA_PATH, item, 'mr_bffe.mhd'))
    array_mr = sitk.GetArrayFromImage(image_mr)
    image_prostaat = sitk.ReadImage(os.path.join(DATA_PATH, item, 'prostaat.mhd'))
    array_prostaat = sitk.GetArrayFromImage(image_prostaat)

    ax[i].imshow(array_mr[50,:,:], cmap='gray')
    ax[i+1].imshow(array_prostaat[50,:,:], cmap='gray')
