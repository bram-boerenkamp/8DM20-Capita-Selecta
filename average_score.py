#%%
#all imports
import elastix
import imageio
import os
import matplotlib.pyplot as plt
import numpy
import SimpleITK as sitk 
import pathlib
import shutil

#global paths
FOLDER_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA'
DATA_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA\TrainingData'
CODE_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA\Code'
#elastix paths and definitions
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
TRANSFORMIX_PATH = os.path.join(FOLDER_PATH, 'transformix.exe')
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
#global variables
patient_list = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120','p125', 'p127', 'p128', 'p129', 'p133', 'p135']
parameters_file_path = os.path.join(CODE_PATH, 'Par0001rigid.txt')
#%% create a result dir and removes all old results!

#%%
#delete all old stuff and make new dirs
output_dir = os.path.join(FOLDER_PATH, 'results_average_score')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
pathlib.Path.mkdir(output_dir, exist_ok=False)

#loop over all patients and register their images
for index_fixed, fixed_patient in enumerate(patient_list):
    for index_moving, moving_patient in enumerate(patient_list):
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
            #%%
            #lets transform the masks!
            tr = elastix.TransformixInterface(parameters=os.path.join(output_dir, 'TransformParameters.0.txt'),
                                              transformix_path=TRANSFORMIX_PATH)
            
            #create output dir and store the moved mask there
            output_dir_mask = os.path.join(output_dir, fixed_patient, moving_patient)
            if os.path.exists(output_dir_mask):
                shutil.rmtree(output_dir_mask)
            output_dir_mask_path = pathlib.Path(output_dir_mask)
            pathlib.Path.mkdir(output_dir_mask_path, parents=True, exist_ok=False)
            tr.transform_image(moving_mask_path, 
                               output_dir=output_dir_mask)
        #else:
        #    continue

#%% 



"""
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

fixed_image_path = os.path.join(CHEST_PATH, 'fixed_image.mhd')
moving_image_path = os.path.join(CHEST_PATH, 'moving_image.mhd')
parameter_file_path = os.path.join(CHEST_PATH, 'parameterswithpenalty.txt')

fixed_image = sitk.ReadImage(fixed_image_path)
moving_image = sitk.ReadImage(moving_image_path)
fixed_image = sitk.GetArrayFromImage(fixed_image)
moving_image = sitk.GetArrayFromImage(moving_image)

el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[parameter_file_path],
    output_dir=os.path.join(FOLDER_PATH, 'results'))
"""



