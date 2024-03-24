#%%
import random
from pathlib import Path

import torch
import u_net
import u_net_utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

# to ensure reproducible training/validation split
random.seed(42)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd().parent.parent / "TrainingData"
DATA_DIR_GEN = Path.cwd().parent.parent / "Generated_images_OG"
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]  # images are made smaller to save training time
GEN_FRAC = 0.5 # possible values are 0.5, 1 and 1.5
BATCH_SIZE = 48 # 48, 64, 96
N_EPOCHS = 50
LEARNING_RATE = 1e-4
TOLERANCE = 0.05  # for early stopping

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


# find patient folders in training directory of generated images
gen_patients = [
    path_gen
    for path_gen in DATA_DIR_GEN.glob("*")
    if not any(part.startswith(".") for part in path_gen.parts)
]
random.shuffle(gen_patients)

# split in training/validation after shuffling
gen_partition = {
    "train": gen_patients[:-NO_VALIDATION_PATIENTS],
    "validation": gen_patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling
dataset = u_net_utils.ProstateMRDataset_with_gen(paths=partition["train"], 
                                                 gen_paths=gen_partition["train"], 
                                                 img_size=IMAGE_SIZE, gen_frac=GEN_FRAC)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = u_net_utils.ProstateMRDataset_with_gen(paths=partition["validation"], 
                                                       gen_paths=gen_partition["train"],
                                                       img_size= IMAGE_SIZE, gen_frac=GEN_FRAC)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

loss_function = u_net_utils.DiceBCELoss()
unet_model = u_net.UNet(num_classes=1).to(device)
optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)

minimum_valid_loss = 1000  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0

    # training iterations
    # tqdm is for timing iteratiions
    for inputs, labels in tqdm(dataloader, position=0):
        # needed to zero gradients in each iterations
        optimizer.zero_grad()
        outputs = unet_model(inputs.to(device))  # forward pass
        loss = loss_function(outputs, labels.float().to(device))
        loss.backward()  # backpropagate loss
        current_train_loss += loss.item()
        optimizer.step()  # update weights
