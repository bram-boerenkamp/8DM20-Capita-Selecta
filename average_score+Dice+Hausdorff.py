#%%
#all imports
import elastix
import os
import matplotlib.pyplot as plt
import numpy
import SimpleITK as sitk 
import pathlib
import shutil
import numpy as np
import scipy
from sklearn import preprocessing

#global paths
FOLDER_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20'
DATA_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20\8DM20-main\TrainingData'
CODE_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20'
#elastix paths and definitions
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
TRANSFORMIX_PATH = os.path.join(FOLDER_PATH, 'transformix.exe')
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

#global variables
patient_list = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120','p125', 'p127', 'p128', 'p129', 'p133', 'p135']
number_of_patients = 3 # for earlier stopping of the code
parameters_file_path = os.path.join(CODE_PATH,'parameter_files',  'Par0001affine.txt')
#%% create a result dir and removes all old results!

#%% registration and transformation
#delete all old stuff and make new dirs
output_dir = os.path.join(FOLDER_PATH, 'results_average_score_ImageTry')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
pathlib.Path.mkdir(output_dir, exist_ok=False)
output_dir_masks = os.path.join(output_dir, 'masks')
pathlib.Path.mkdir(output_dir_masks, exist_ok=False) # make a dir for the masks

#%%
#loop over all patients and register their images
for index_fixed, fixed_patient in enumerate(patient_list):
    for index_moving, moving_patient in enumerate(patient_list):
        if number_of_patients > index_fixed:
            if index_fixed != index_moving:
                #define and read moving and fixed images and masks
                fixed_image_path = os.path.join(DATA_PATH, fixed_patient, 'mr_bffe.mhd')
                moving_image_path = os.path.join(DATA_PATH, moving_patient, 'mr_bffe.mhd')
                fixed_mask_path = os.path.join(DATA_PATH, fixed_patient, 'prostaat.mhd')
                moving_mask_path = os.path.join(DATA_PATH, moving_patient, 'prostaat.mhd')

                fixed_image = sitk.ReadImage(fixed_image_path)
                moving_image = sitk.ReadImage(moving_image_path)
                fixed_image = sitk.GetArrayFromImage(fixed_image)
                moving_image = sitk.GetArrayFromImage(moving_image)
                #lets register!
                el.register(
                            fixed_image=fixed_image_path,
                            moving_image=moving_image_path,
                            parameters=[parameters_file_path],
                            output_dir=output_dir
                            )
                
                #lets transform the masks!
                tr = elastix.TransformixInterface(parameters=os.path.join(output_dir, 'TransformParameters.0.txt'),
                                                transformix_path=TRANSFORMIX_PATH)
                
                #create output dir and store the moved mask there
                output_dir_mask = os.path.join(output_dir_masks, fixed_patient, moving_patient)
                if os.path.exists(output_dir_mask):
                    shutil.rmtree(output_dir_mask)
                output_dir_mask_path = pathlib.Path(output_dir_mask)
                pathlib.Path.mkdir(output_dir_mask_path, parents=True, exist_ok=False)
                tr.transform_image(moving_mask_path, output_dir=output_dir_mask)
#%% calculating average masks per patient
                
for fixed_mask in os.listdir(output_dir_masks):
    # bring all moved masks values to one mask
    average_mask = numpy.empty((86, 333, 271), dtype='int16') #initialization array
    for i, moving_mask in enumerate(os.listdir(os.path.join(output_dir_masks, fixed_mask))):
        moving_mask_imagefile = os.path.join(output_dir_masks, fixed_mask, moving_mask, 'result.mhd')
        moving_mask_image = sitk.ReadImage(moving_mask_imagefile)
        pixel_values = sitk.GetArrayFromImage(moving_mask_image)
        average_mask += pixel_values
    new_mask = sitk.GetImageFromArray(average_mask)
    
    #make mask binary again
    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(new_mask)
    threshold = filter.GetMaximum()/2

    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(0)
    binary_filter.SetUpperThreshold(threshold)
    binary_filter.SetInsideValue(0)
    binary_filter.SetOutsideValue(1)

    binary_mask = binary_filter.Execute(new_mask)
    binary_mask_array = sitk.GetArrayFromImage(binary_mask) 

    #show some images
    fixed_mask_dir = os.path.join(DATA_PATH, fixed_mask, 'prostaat.mhd')
    fixed_mask_image = sitk.ReadImage(fixed_mask_dir)
    fixed_mask_array = sitk.GetArrayFromImage(fixed_mask_image)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].imshow(average_mask[50,:,:], cmap='gray')
    ax[0].set_title('all tranformed masks')
    ax[1].imshow(binary_mask_array[50,:,:], cmap='gray')
    ax[1].set_title('binary mask of all transformed masks')
    ax[2].imshow(fixed_mask_array[50,:,:], cmap='gray')
    ax[2].set_title('fixed mask')
    plt.show()
   
#%% calculate dice score per patient and average dice score

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
            transformed_mask_path = os.path.join(output_dir_masks,fixed_patient,transformed_patient,'result.mhd')
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



