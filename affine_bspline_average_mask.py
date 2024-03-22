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


#%%
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

#%%
#global paths
FOLDER_PATH =  r'D:\Documenten\Master\Q3\Capita selecta image analysis\Elastix'
DATA_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Data\blurred_level_2'
CODE_PATH = r'D:\Documenten\Master\Q3\Capita selecta image analysis\Grid size test'
#elastix paths and definitions
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
TRANSFORMIX_PATH = os.path.join(FOLDER_PATH, 'transformix.exe')
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

#global variables
patient_list = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120','p125', 'p127', 'p128', 'p129', 'p133', 'p135']
number_of_patients = 15 # for earlier stopping of the code
parameters_file_path_affine = os.path.join(CODE_PATH,'parameter_files',  'Par0001affine(new).txt')
parameters_file_path_bspline = os.path.join(CODE_PATH,'parameter_files',  'Parameter_file_bspline_final.txt')
#%% create a result dir and removes all old results!

#%% registration and transformation
#delete all old stuff and make new dirs
output_dir_1 = os.path.join(FOLDER_PATH, 'results_affine_blurred_level_2')
output_dir_1 = pathlib.Path(output_dir_1)
output_dir_2 = os.path.join(FOLDER_PATH, 'results_affine_bspline_blurred_level_2')
output_dir_2 = pathlib.Path(output_dir_2)
if os.path.exists(output_dir_1):
    shutil.rmtree(output_dir_1)
if os.path.exists(output_dir_2):
    shutil.rmtree(output_dir_2)
pathlib.Path.mkdir(output_dir_1, exist_ok=False)
pathlib.Path.mkdir(output_dir_2, exist_ok=False)
output_dir_masks_1 = os.path.join(output_dir_1, 'masks')
output_dir_masks_1 = pathlib.Path(output_dir_masks_1)
output_dir_masks_2 = os.path.join(output_dir_2,'masks')
output_dir_masks_2 = pathlib.Path(output_dir_masks_2)
pathlib.Path.mkdir(output_dir_masks_1, exist_ok=False) # make a dir for the masks
pathlib.Path.mkdir(output_dir_masks_2, exist_ok=False) 

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
                            parameters=[parameters_file_path_affine],
                            output_dir=str(output_dir_1)
                            )
                
                #lets transform the masks!
                tr = elastix.TransformixInterface(parameters=os.path.join(output_dir_1, 'TransformParameters.0.txt'),
                                                transformix_path=TRANSFORMIX_PATH)
                
                #create output dir and store the moved mask there
                output_dir_mask_1 = os.path.join(output_dir_masks_1, fixed_patient, moving_patient)
                if os.path.exists(output_dir_mask_1):
                    shutil.rmtree(output_dir_mask_1)
                output_dir_mask_path_1 = pathlib.Path(output_dir_mask_1)
                pathlib.Path.mkdir(output_dir_mask_path_1, parents=True, exist_ok=False)
                tr.transform_image(moving_mask_path, output_dir=output_dir_mask_1)
                
                
                #use the result of the affine transformation as input for the bspline
                moving_mask_path_2 = os.path.join(output_dir_masks_1,fixed_patient,moving_patient,'result.mhd')
                moving_image_path_2 = os.path.join(output_dir_1, 'result.0.mhd')
                
                #%%

                el.register(
                            fixed_image=fixed_image_path,
                            moving_image=moving_image_path_2,
                            parameters=[parameters_file_path_bspline],
                            output_dir=str(output_dir_2))
                
                #%%
                #lets transform the masks!
                tr = elastix.TransformixInterface(parameters=os.path.join(output_dir_2, 'TransformParameters.0.txt'),
                                                transformix_path=TRANSFORMIX_PATH)
                
                #create output dir and store the moved mask there
                output_dir_mask_2 = os.path.join(output_dir_masks_2, fixed_patient, moving_patient)
                if os.path.exists(output_dir_mask_2):
                    shutil.rmtree(output_dir_mask_2)
                output_dir_mask_path_2 = pathlib.Path(output_dir_mask_2)
                pathlib.Path.mkdir(output_dir_mask_path_2, parents=True, exist_ok=False)
                tr.transform_image(moving_mask_path_2, output_dir=output_dir_mask_2)
                

#%% calculating average masks per patient

#initialize the dice score and hausdorff distance
dice_scores = []
hausdorff_distances = []
                
for fixed_mask in os.listdir(output_dir_masks_2):
    # bring all moved masks values to one mask
    average_mask = numpy.empty((86, 333, 271), dtype='int16') #initialization array
    for i, moving_mask in enumerate(os.listdir(os.path.join(output_dir_masks_2, fixed_mask))):
        moving_mask_imagefile = os.path.join(output_dir_masks_2, fixed_mask, moving_mask, 'result.mhd')
        moving_mask_image = sitk.ReadImage(moving_mask_imagefile)
        pixel_values = sitk.GetArrayFromImage(moving_mask_image)
        average_mask += pixel_values
    new_mask = sitk.GetImageFromArray(average_mask)
    
    #make mask binary again
    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(new_mask)
    threshold = filter.GetMaximum()*(10/15)

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
    
    #Calculate the dice score between the generated and the fixed mask
    dice_score = dice(binary_mask_array,fixed_mask_array)
    dice_scores.append(dice_score)
    
    #Calculate the Hausdorff distance between the generated and fixed mask 
    fixed_points = np.transpose(np.nonzero(fixed_mask_array))
    transformed_points = np.transpose(np.nonzero(binary_mask_array))

    # Directed Hausdorff distance, first value is the HD, second and third value are the point that corresponds with the HD
    hausdorff_distance = scipy.spatial.distance.directed_hausdorff(fixed_points, transformed_points)
    hausdorff_distance = hausdorff_distance[0]
    hausdorff_distances.append(hausdorff_distance)
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].imshow(average_mask[50,:,:], cmap='gray')
    ax[0].set_title('all tranformed masks')
    ax[1].imshow(binary_mask_array[50,:,:], cmap='gray')
    ax[1].set_title('binary mask of all transformed masks')
    ax[2].imshow(fixed_mask_array[50,:,:], cmap='gray')
    ax[2].set_title('fixed mask')
    plt.show()
    
    

#%% 
print("{:<10} {:<15} {:<10}".format('Patient', 'Dice score', 'Hausdorff distance'))
for i in range(number_of_patients):
    print("{:<10} {:<15.2f} {:<10.2f}".format(patient_list[i], dice_scores[i], hausdorff_distances[i]))
   



