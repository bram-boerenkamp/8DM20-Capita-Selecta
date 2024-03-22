import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ProstateMRDataset_with_gen(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, gen_paths, img_size, gen_frac=1):
        self.mr_image_list = []
        self.mask_list = []
        self.gen_list = []
        self.len_paths = len(paths)
        self.len_gen_paths = len(gen_paths)
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd")).astype(
                    np.int32
                )
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(
                    np.int32
                )
            )
        if gen_frac == 1: #50/50 datasplit real/fake
            for path in gen_paths:
                self.gen_list.append(
                    sitk.GetArrayFromImage(sitk.ReadImage(path / "generated.mhd")).astype(
                        np.float32
                    )
                )
                self.mask_list.append(
                    sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(
                        np.int32
                    )
                )  

        if gen_frac == 0.5: #72/25 datasplit real/fake
            for path in gen_paths:
                if len(self.gen_list) < len(self.mr_image_list)/2:
                    self.gen_list.append(
                        sitk.GetArrayFromImage(sitk.ReadImage(path / "generated.mhd")).astype(
                            np.float32
                        )
                    )
                    self.mask_list.append(
                        sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(
                            np.int32
                        )
                    )  
        if gen_frac == 1.5: # 25/75 datasplit real/fake
            for i in range(1):
                for path in gen_paths:
                    self.gen_list.append(
                        sitk.GetArrayFromImage(sitk.ReadImage(path / "generated.mhd")).astype(
                            np.float32
                        )
                    )
                    self.mask_list.append(
                        sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(
                            np.int32
                        )
                    )  
        
        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)+len(self.gen_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to resize images
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(mode="I"),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )

        self.gen_img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.

        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)
        if patient <= self.len_paths-1:
            return (
                self.norm_transform(
                    self.img_transform(self.mr_image_list[patient][the_slice, ...]).float()
                ),
                self.img_transform(
                    (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
                ),
            )
        else: 
            return(

                    self.gen_img_transform(self.gen_list[patient//2][the_slice, ...]).float()
                ,
                self.img_transform(
                    (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
                ),
            )

class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size):
        self.mr_image_list = []
        self.mask_list = []
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd")).astype(
                    np.int32
                )
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(
                    np.int32
                )
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to resize images
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(mode="I"),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )
        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.

        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        return (
            self.norm_transform(
                self.img_transform(self.mr_image_list[patient][the_slice, ...]).float()
            ),
            self.img_transform(
                (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
            ),
        )


class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1

        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            outputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return BCE + dice_loss