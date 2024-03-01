#%%
#all imports
import elastix
import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk 
import pathlib
import shutil
import scipy
from sklearn import preprocessing

#global paths
FOLDER_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Elastix'
DATA_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Data\TrainingData'
CODE_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Hausdorff and dice score'
TRANSFORMED_MASKS_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Elastix\results_average_score\masks'

patient_list = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120','p125', 'p127', 'p128', 'p129', 'p133', 'p135']
#Select for how many patients you want to compute the similarity indices
number_of_patients = 1

dice_scores = {}
hausdorff_distances = {}
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

#Loop through the patients
for index_fixed, fixed_patient in enumerate(patient_list):
    if number_of_patients > index_fixed:
        #Get the fixed initial mask 
        fixed_mask_path = os.path.join(DATA_PATH, fixed_patient, 'prostaat.mhd')
        fixed_mask = sitk.ReadImage(fixed_mask_path)
        fixed_mask = sitk.GetArrayFromImage(fixed_mask)
        
        #Specify the path of the transformed masks results
        subfolder_transformed_masks = [patient for patient in patient_list if patient != fixed_patient]
        #Loop through the transformed masks folder
        for index_transformed, transformed_patient in enumerate(subfolder_transformed_masks):
            transformed_mask_path = os.path.join(TRANSFORMED_MASKS_PATH,fixed_patient,transformed_patient,'result.mhd')
            transformed_mask = sitk.ReadImage(transformed_mask_path)
            transformed_mask = sitk.GetArrayFromImage(transformed_mask)
            
            #Calculate the dice score
            dice_score = dice(fixed_mask,transformed_mask)
            pair_key = (fixed_patient, transformed_patient)
            dice_scores[pair_key] = dice_score
            
            #Calculate the Hausdorff distance
            fixed_points = np.transpose(np.nonzero(fixed_mask))
            transformed_points = np.transpose(np.nonzero(transformed_mask))

            # Directed Hausdorff distance, first value is the HD, second and third value are the point that corresponds with the HD
            hausdorff_distance = scipy.spatial.distance.directed_hausdorff(fixed_points, transformed_points)
            hausdorff_distance = hausdorff_distance[0]
            hausdorff_distances[pair_key] = hausdorff_distance
            
            #%%
            #Get the weights for each mask
            normalized_weights_hausdorff = []
            normalized_weights_dice = []
            normalized_hausdorff = preprocessing.normalize([list(hausdorff_distances.values())])
            #Because a high hausdorff indicates a bad mask, convert this metric by subtracting by 1
            normalized_hausdorff = 1-normalized_hausdorff
            normalized_dice = preprocessing.normalize([list(dice_scores.values())])
            average_weights_dice_and_hd = []
            for i in range((normalized_dice.shape[1])):
                average_weights_dice_and_hd.append((normalized_hausdorff[0,i]+normalized_dice[0,i])/2)


                
        





