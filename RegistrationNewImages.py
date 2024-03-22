#%%
#all imports
import elastix
import os
import numpy
import SimpleITK as sitk 
import pathlib
import shutil


#%%
#global paths
FOLDER_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20'
DATA_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20\8DM20-main\TrainingData'
CODE_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20'

#elastix paths and definitions
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
TRANSFORMIX_PATH = os.path.join(FOLDER_PATH, 'transformix.exe')
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

#global variables
patient_list = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120','p125']
NewPatient_list = ['p227', 'p228', 'p229', 'p233', 'p235']
#number_of_patients = 15 # for earlier stopping of the code
parameters_file_path_affine = os.path.join(CODE_PATH,'parameter_files',  'Par0001affine(new).txt')
parameters_file_path_bspline = os.path.join(CODE_PATH,'parameter_files',  'Parameter_file_bspline_final.txt')
#%% create a result dir and removes all old results!

#%% registration and transformation
#delete all old stuff and make new dirs
output_dir_1 = os.path.join(FOLDER_PATH, 'results_affine')
output_dir_1 = pathlib.Path(output_dir_1)
output_dir_2 = os.path.join(FOLDER_PATH, 'results_affine_bspline')
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
for index_fixed, fixed_patient in enumerate(NewPatient_list):
    for index_moving, moving_patient in enumerate(patient_list):
        #define and read moving and fixed images and masks
        fixed_image_path = os.path.join(DATA_PATH, fixed_patient, 'mr_bffe.mhd')
        moving_image_path = os.path.join(DATA_PATH, moving_patient, 'mr_bffe.mhd')
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
            shutil.rmtree(output_dir_mask_2)
        output_dir_mask_path_2 = pathlib.Path(output_dir_mask_2)
        pathlib.Path.mkdir(output_dir_mask_path_2, parents=True, exist_ok=False)
        tr.transform_image(moving_mask_path_2, output_dir=output_dir_mask_2)
        

#%% Creating output direction for average masks

output_dir_average = os.path.join(FOLDER_PATH, 'results_Average/')
if os.path.exists(output_dir_average):
    shutil.rmtree(output_dir_average)
pathlib.Path.mkdir(output_dir_average, exist_ok = False)

#%% calculating average masks per patient                
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
    #Save average masks
    img = sitk.GetImageFromArray(binary_mask_array)
    sitk.WriteImage(img, output_dir_average + str(fixed_mask) + "AverageMask.mhd")

    

    


