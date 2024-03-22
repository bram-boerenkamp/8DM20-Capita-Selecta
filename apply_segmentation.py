import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from pathlib import Path
from skimage.metrics import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure
from skimage.measure import regionprops

import u_net
import utils
import cv2
import SimpleITK as sitk

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd().parent / r"C:\Users\Nitikaa\Downloads\2ndpartcapita\Data"
BEST_EPOCH = 49
CHECKPOINTS_DIR = Path.cwd() / r"C:\Users\Nitikaa\Downloads\2ndpartcapita\segmentation_model_weights" / f"u_net_{BEST_EPOCH}.pth"

# hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)

unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
unet_model.eval()

image = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\Nitikaa\Downloads\2ndpartcapita\Data\p102\mr_bffe.mhd")).astype(np.int32)
output = np.zeros((image.shape[0], 256, 256))  # Corrected line

for slice_ind in range(image.shape[0]):
    slice_img = image[slice_ind]  # Renamed to slice_img to avoid confusion with Python's slice
    slice_img = transforms.ToPILImage(mode="I")(slice_img)
    slice_img = transforms.CenterCrop(256)(slice_img)
    slice_img = transforms.PILToTensor()(slice_img)
    
    # Assuming your model expects a batch dimension and a channel dimension
    # Add batch dimension with .unsqueeze(0) and channel dimension with .unsqueeze(0)
    slice_img = slice_img.unsqueeze(0).float()  # Add only the batch dimension
    pred = unet_model(slice_img).detach().cpu().numpy()  # Ensure tensor is moved to cpu
    
    output[slice_ind, :, :] = pred[0, 0, :, :]
    
mask = np.zeros(output.shape)
mask[output < -3000] = 1  # have set threshold to plot it in binary instead of a heat map
plt.imshow(mask[45], cmap="gray")
plt.colorbar()
plt.show()

from skimage.transform import resize
from skimage.metrics import hausdorff_distance

def dice_coefficient(predicted, target):
    # Calculate the Dice Coefficient
    intersection = np.sum(predicted * target)
    union = np.sum(predicted) + np.sum(target)
    dice_coef = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice_coef

# Load the ground truth mask
ground_truth_mask = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\Nitikaa\Downloads\2ndpartcapita\Data\p102\prostaat.mhd")).astype(np.int32)

# Choose the 45th slice from your output mask
predicted_slice = mask[45]

# Choose the 45th slice from the ground truth mask and resize it to match the predicted slice shape
ground_truth_slice = ground_truth_mask[45]
ground_truth_slice_resized = resize(ground_truth_slice, predicted_slice.shape, preserve_range=True).astype(np.int32)

# Ensure that the ground truth mask is binary
ground_truth_slice_binary = np.zeros_like(ground_truth_slice_resized)
ground_truth_slice_binary[ground_truth_slice_resized < -3000] = 0  # Assuming same thresholding applies
ground_truth_slice_binary[ground_truth_slice_resized >= -3000] = 1

# Compute Dice Coefficient
dice_score = dice_coefficient(predicted_slice, ground_truth_slice_binary)
print(f"Dice Score: {dice_score}")

# Compute Hausdorff Distance
# Note: The Hausdorff distance function from skimage.metrics expects boolean arrays
hausdorff_dist = hausdorff_distance(predicted_slice.astype(bool), ground_truth_slice_binary.astype(bool))
print(f"Hausdorff Distance: {hausdorff_dist}")



# TODO
# apply for all images and compute Dice score with ground-truth.
# output .mhd images with the predicted segmentations
with torch.no_grad():
    predict_index = 45  # here I just chose a random slice for testing
    # you should do this for all slices
    (input, target) = valid_dataset[predict_index]
    output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
    prediction = torch.round(output)
    
   

    plt.subplot(131)
    plt.imshow(input[0], cmap="gray")
    plt.subplot(132)
    plt.imshow(target[0])
    plt.subplot(133)
    plt.imshow(prediction[0, 0])
    plt.show()
