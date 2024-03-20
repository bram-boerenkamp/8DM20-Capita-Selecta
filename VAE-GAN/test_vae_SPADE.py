#%%
import random
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import SimpleITK as sitk

import vae_SPADE
import utils
import os
import shutil

# to ensure reproducible training/validation split
random.seed(42)

# directories with data and to stored training checkpoints
DATA_DIR = Path.cwd().parent / "TrainingData"
DATA_DIR_GEN = Path.cwd().parent / "Generated_images"
if os.path.exists(DATA_DIR_GEN):
    shutil.rmtree(DATA_DIR_GEN)
DATA_DIR_GEN.mkdir(parents=True, exist_ok=True)
BEST_EPOCH = 299 # set this by hand by visually inspecting your best epoch in tensorboard
CHECKPOINTS_DIR = Path.cwd() / "vae_SPADE_model_weights_no_lrsh" / f"vae_SPADE_{BEST_EPOCH}.pth"
patient_list = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120','p125', 'p127', 'p128', 'p129', 'p133', 'p135']

# hyperparameters
NO_VALIDATION_PATIENTS = 15
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

vae_SPADE_model = vae_SPADE.VAE_SPADE()
vae_SPADE_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
vae_SPADE_model.eval()
vae_SPADE_model.cpu()



patient = 0
labels_list = []
outputs_list = []

with torch.no_grad():

    for slice in range(len(valid_dataset)):
        inputs, targets  = valid_dataset[slice]
        outputs, mu, logvar = vae_SPADE_model(inputs[np.newaxis, ...], targets[np.newaxis, ...])

        labels_list.append(targets[0])
        outputs_list.append(outputs[0,0])

        if (slice+1) % 86 == 0:
            labels = np.stack(labels_list)
            outputs = np.stack(outputs_list)

            labels_itk = sitk.GetImageFromArray(labels.astype(np.float32))
            outputs_itk = sitk.GetImageFromArray(outputs.astype(np.float32))

            patient_path = os.path.join(DATA_DIR_GEN, patient_list[patient])
            patient_path = Path(patient_path)
            patient_path.mkdir(parents=True, exist_ok=True)

            output_path = os.path.join(DATA_DIR_GEN, patient_list[patient], 'generated.mhd')
            label_path = os.path.join(DATA_DIR_GEN, patient_list[patient], 'prostaat.mhd')

            sitk.WriteImage(outputs_itk, output_path)
            sitk.WriteImage(labels_itk, label_path)
            labels_list = []
            outputs_list = []
            patient += 1
            plt.subplot(131)
            plt.imshow(inputs[0], cmap="gray")
            plt.subplot(132)
            plt.imshow(targets[0], cmap='grey')
            plt.subplot(133)
            plt.imshow(outputs[0], cmap='grey')
            plt.show()

vae_SPADE_model.cuda()
vae_SPADE_model.train()
