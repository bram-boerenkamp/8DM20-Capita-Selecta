#%%
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchinfo import summary 
from torchvision.utils import save_image
import utils
import vae_SPADE

# to ensure reproducible training/validation split
random.seed(42)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#%%
# directories with data and to store training checkpoints and logs
DATA_DIR = Path.cwd().parent.parent / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "no_noise_VAE_SPADE_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "no_noise_VAE_SPADE_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 300
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 1
SAVE_FREQ = 10
TOLERANCE = 0.05  # for early stopping

# dimension of VAE latent space
Z_DIM = 256

# function to reduce the learning rate
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )


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

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser
vae_SPADE_model = vae_SPADE.VAE_SPADE().to(device) 

#####################################################################
# function to show a model summary use verbose =2 for more information
# can help a lot to check if the model does what you intend it to do
#####################################################################
#print(summary(vae_SPADE_model, verbose =2))  

optimizer = torch.optim.Adam(vae_SPADE_model.parameters(), lr=LEARNING_RATE) 
# add a learning rate scheduler based on the lr_lambda function
#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda) # scheduler based on the lr_lambda function

minimum_valid_loss = 1000 
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0

    for inputs, labels in tqdm(dataloader, position=0):
        optimizer.zero_grad()
        outputs, mu, logvar = vae_SPADE_model(inputs.to(device), labels.to(device)) #Putting input and labels to get the output
        loss = vae_SPADE.vae_SPADE_loss(inputs=inputs.to(device),
                            recons=outputs.to(device),
                            mu=mu.to(device),
                            logvar=logvar.to(device)) #update the weights
        loss.backward() # backpropagate the weights to update them
        optimizer.step() #calling the optimizer 
        current_train_loss += loss.item() #record the training loss 



    # evaluate validation loss
    with torch.no_grad():
        vae_SPADE_model.eval()
        for inputs, labels in tqdm(valid_dataloader, position=0):
            outputs, mu, logvar = vae_SPADE_model(inputs.to(device), labels.to(device)) #Putting input to get the output
            loss = vae_SPADE.vae_SPADE_loss(inputs=inputs.to(device),
                            recons=outputs.to(device),
                            mu=mu.to(device),
                            logvar=logvar.to(device))
            current_valid_loss += loss.item()
        vae_SPADE_model.train()

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )
    #scheduler.step() # step the learning step scheduler

    # save examples of real/fake images
    if (epoch + 1) % DISPLAY_FREQ == 0:
        vae_SPADE_model.eval()
        vae_SPADE_model.cpu()
        x_real = inputs
        x_recon, mu, logvar = vae_SPADE_model(inputs, labels)
        img_grid = make_grid(
            torch.cat((x_recon[:5], x_real[:5])), nrow=5, padding=12, pad_value=-1
        )
        writer.add_image(
            "Real_fake", np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5, epoch + 1
        )
        vae_SPADE_model.cuda()
        vae_SPADE_model.train()

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        weights_dict = {k: v.cpu() for k, v in vae_SPADE_model.state_dict().items()}
        if epoch > 200:
            if (epoch + 1) % SAVE_FREQ == 0:
                torch.save(
                    weights_dict,
                    CHECKPOINTS_DIR / f"VAE_SPADE_{epoch}.pth",
                )
#%%    
weights_dict = {k: v.cpu() for k, v in vae_SPADE_model.state_dict().items()}
torch.save(weights_dict, CHECKPOINTS_DIR / f"VAE_SPADE_{epoch}.pth")
#%%
print(partition["validation"])