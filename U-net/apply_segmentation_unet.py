#%%
import os
import random
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff

import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

import u_net
import u_net_utils
import SimpleITK as sitk
import shutil

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints

# Change NEW_IMAGES to True for the 5 new patients and change the DATA_DIR to the ValidateData
# or used NEW_IMAGES = False for the 15 old ones and change DATA_DIR to TrainingData 
NEW_IMAGES = False
DATA_DIR = Path.cwd().parent.parent / "ValidateData"
SEGMENTATION_DIR = Path.cwd().parent.parent / "Predicted_masks"
if os.path.exists(SEGMENTATION_DIR):
    shutil.rmtree(SEGMENTATION_DIR)
    SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
DOWNSAMPLED_DIR = Path.cwd().parent.parent / "Downsampled_TrainingData"
if os.path.exists(DOWNSAMPLED_DIR):
    shutil.rmtree(DOWNSAMPLED_DIR)
DOWNSAMPLED_DIR.mkdir(parents=True, exist_ok=True)

BEST_EPOCH = 34
CHECKPOINTS_DIR = Path.cwd() / "no_noise_gen_unet_model_weights" / f"u_net_{BEST_EPOCH}.pth"
patient_list = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120','p125', 'p127', 'p128', 'p129', 'p133', 'p135']
new_patient_list = ['p118', 'p145', 'p146', 'p149', 'p150']

# hyperparameters
NO_VALIDATION_PATIENTS = 15
IMAGE_SIZE = [64, 64]



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

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load validation data
if NEW_IMAGES == True:
    valid_dataset = u_net_utils.ProstateMRDataset_for_new(partition["validation"], IMAGE_SIZE)
if NEW_IMAGES == False:
    valid_dataset = u_net_utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE) 

unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
unet_model.eval()
unet_model.cpu()

patient = 0
outputs_list = []
labels_list = []
inputs_list = []
dice_scores = []
hausdorff_distances = []
'''
    output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
    prediction = torch.round(output)
'''
with torch.no_grad():
    for slice in range(len(valid_dataset)):
        if NEW_IMAGES == True:
            inputs = valid_dataset[slice]
            outputs = torch.sigmoid(unet_model(inputs[np.newaxis, ...]))
            outputs = torch.round(outputs)
        if NEW_IMAGES == False:
            inputs, targets = valid_dataset[slice]
            outputs = torch.sigmoid(unet_model(inputs[np.newaxis, ...]))
            outputs = torch.round(outputs)

        if NEW_IMAGES ==False:
            labels_list.append(targets[0])
        outputs_list.append(outputs[0,0])
        inputs_list.append(inputs[0])

        if (slice+1) % 86 == 0:
            if NEW_IMAGES == False:
                labels_array = np.stack(labels_list)
                labels_itk = sitk.GetImageFromArray(labels_array)
            
            outputs_array = np.stack(outputs_list)
            outputs_itk = sitk.GetImageFromArray(outputs_array)

            inputs_array = np.stack(inputs_list)
            inputs_itk = sitk.GetImageFromArray(inputs_array)
            

            if NEW_IMAGES == True:
                patient_path = os.path.join(SEGMENTATION_DIR, new_patient_list[patient])
                downsampled_path = os.path.join(DOWNSAMPLED_DIR, new_patient_list[patient])
            if NEW_IMAGES == False:
                patient_path = os.path.join(SEGMENTATION_DIR, patient_list[patient])
                downsampled_path = os.path.join(DOWNSAMPLED_DIR, patient_list[patient])
            patient_path = Path(patient_path)
            downsampled_path = Path(downsampled_path)
            patient_path.mkdir(parents=True, exist_ok=True)
            downsampled_path.mkdir(parents=True, exist_ok=True)

            if NEW_IMAGES == True:
                output_path = os.path.join(SEGMENTATION_DIR, new_patient_list[patient], 'prostaat.mhd')
                output_downsample_path = os.path.join(DOWNSAMPLED_DIR, new_patient_list[patient], 'downsampled.mhd')
            if NEW_IMAGES == False:
                output_path = os.path.join(SEGMENTATION_DIR, patient_list[patient], 'prostaat.mhd')
                output_downsample_path = os.path.join(DOWNSAMPLED_DIR, patient_list[patient], 'downsampled.mhd')
            
            sitk.WriteImage(outputs_itk, output_path)
            sitk.WriteImage(inputs_itk, output_downsample_path)

            plt.subplot(131)
            plt.imshow(inputs_list[42], cmap="gray")
            plt.title('input image')
            plt.subplot(132)
            plt.imshow(outputs_list[42], cmap='grey')
            plt.title('output mask')
            plt.subplot(133)
            if NEW_IMAGES == False:
                plt.imshow(labels_list[42], cmap='grey')
                plt.title('original mask')
            plt.show()

            if NEW_IMAGES == False:
                dice_score = dice(outputs_array, labels_array)
                dice_scores.append(dice_score)


                output_points = np.transpose(np.nonzero(outputs_array))
                labels_points = np.transpose(np.nonzero(labels_array))            

                hausdorff_distance = directed_hausdorff(output_points, labels_points)
                hausdorff_distances.append(hausdorff_distance[0])
                labels_list = []
            outputs_list = []
            inputs_list = []
            patient += 1

unet_model.cuda()
unet_model.train()
print("{:<10} {:<15} {:<10}".format('Patient', 'Dice score', 'Hausdorff distance'))
for i in range(len(patients)):
    if NEW_IMAGES == True:
        print("{:<10} {:<15.2f} {:<10.2f}".format(new_patient_list[i], dice_scores[i], hausdorff_distances[i]))
    if NEW_IMAGES == False:
        print("{:<10} {:<15.2f} {:<10.2f}".format(patient_list[i], dice_scores[i], hausdorff_distances[i]))