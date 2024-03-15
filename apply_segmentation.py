import random
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import SimpleITK as sitk
import u_net
import utils
from utils import load_3d_image_as_numpy
from torchvision.transforms import Compose, ToPILImage, CenterCrop, Resize, ToTensor, Normalize
import torch.nn as nn
import torchvision.transforms as transforms
from u_net import UNet
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import interpolate

# to ensure reproducible training/validation split
random.seed(42)

# This is an example; you should replace it with the actual path to your 3D image
three_d_image_path = Path(r'C:\Users\Nitikaa\Downloads\8DM20-main\8DM20-main\code\Data\p102\mr_bffe.mhd')


# directorys with data and to stored training checkpoints
DATA_DIR = Path(r'C:\Users\Nitikaa\Downloads\8DM20-main\8DM20-main\code\Data')
BEST_EPOCH = 49
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights" / f"u_net_{BEST_EPOCH}.pth"

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


image_3d_sitk = sitk.ReadImage(str(three_d_image_path))
image_3d_np = sitk.GetArrayFromImage(image_3d_sitk)

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(256),  # Adjust size as necessary
    transforms.Resize([64, 64]),  # Resize to match U-Net input size
    transforms.ToTensor(),
])

# Calculate the mean and std from the dataset or use hardcoded values from training
mean = 0.0  # Replace with the mean from your dataset
std = 1.0  # Replace with the std from your dataset
norm_transform = transforms.Normalize(mean, std)

#img_transform = utils.ProstateMRDataset.img_transform
#norm_transform = utils.ProstateMRDataset.norm_transform

# Placeholder for full resolution mask
# Assuming the following are defined earlier in your script:
# img_transform - A torchvision transform for preprocessing
# norm_transform - A torchvision transform for normalizing
# unet_model - Your trained U-Net model
full_res_mask = []
original_3d_image_itk = sitk.ReadImage(str(three_d_image_path))
original_size = original_3d_image_itk.GetSize()

for slice_idx in range(image_3d_np.shape[0]):
    slice_2d = image_3d_np[slice_idx, :, :]
    
    # Create a PIL Image from the numpy array
    slice_2d_pil = Image.fromarray(slice_2d)
    
    # If your original image is a single channel (grayscale), you should ensure the PIL image is in 'L' mode
    if slice_2d_pil.mode != 'L':
        slice_2d_pil = slice_2d_pil.convert('L')
    
    # Resize the PIL image using the Resize transform
    slice_2d_resized = TF.resize(slice_2d_pil, [64, 64], interpolation=Image.NEAREST)

    # Convert the resized PIL image to a tensor
    slice_2d_tensor = TF.to_tensor(slice_2d_resized)

    # Normalize the tensor
    slice_2d_normalized = TF.normalize(slice_2d_tensor, [mean], [std]).unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = unet_model(slice_2d_normalized)

    # Apply a sigmoid function and threshold to create a binary mask
    prediction_binary = torch.sigmoid(prediction) > 0.5

    # Convert PyTorch tensor to numpy array and remove the added batch dimension
    prediction_binary_np = prediction_binary.squeeze().cpu().numpy()

    # Append the binary mask of current slice to the list
    full_res_mask.append(prediction_binary_np)
    
final_3d_mask = np.stack(full_res_mask, axis=0)

# Create a SimpleITK image from the numpy array
final_3d_mask_itk = sitk.GetImageFromArray(final_3d_mask.astype(np.float32))

# Now, use SimpleITK to resize the 3D mask to match the original image size
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(original_3d_image_itk)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetOutputSpacing(original_3d_image_itk.GetSpacing())
resampler.SetSize(original_3d_image_itk.GetSize())
resampler.SetTransform(sitk.Transform())
final_3d_mask_resized_itk = resampler.Execute(final_3d_mask_itk)

# Copy the metadata from the original 3D image to the resized 3D mask
final_3d_mask_resized_itk.CopyInformation(original_3d_image_itk)

# Save the SimpleITK image to disk
output_path = r'C:\Users\Nitikaa\Downloads\8DM20-main\8DM20-main\code\output.mhd'  # Specify your output file path here
sitk.WriteImage(final_3d_mask_resized_itk, output_path)

# TODO
# apply for all images and compute Dice score with ground-truth.
# output .mhd images with the predicted segmentations
with torch.no_grad():
    predict_index = 75  # here I just chose a random slice for testing
    # you should do this for all slices
    (input, target) = valid_dataset[predict_index]
    output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
    prediction = torch.round(output)
'''
    plt.subplot(131)
    plt.imshow(input[0], cmap="gray")
    plt.subplot(132)
    plt.imshow(target[0])
    plt.subplot(133)
    plt.imshow(prediction[0, 0])
    plt.show()
'''