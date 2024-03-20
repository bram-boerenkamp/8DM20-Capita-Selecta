import torch
import torch.nn as nn


class Block(nn.Module):
    """A representation for the basic convolutional building block of the unet

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        """Performs the forward pass of the block.

        Parameters
        ----------
        x : torch.Tensor
            the input to the block

        Returns
        -------
        x : torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """A representation for the encoder part of the unet.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the encoder

    """

    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        list[torch.Tensor]
            contains the outputs of each block in the encoder
        """
        ftrs = []  # a list to store features
        for block in self.enc_blocks:
            x = block(x)
            # # save features to concatenate to decoder blocks
            ftrs.append(x)
            x = self.pool(x)
        ftrs.append(x)  # save features
        return ftrs


class Decoder(nn.Module):
    """A representation for the decoder part of the unet.

    Layers consist of transposed convolutions followed by convolutional blocks.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the decoder
    """

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(2 * chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )  # the first argument of the Block is multipled by 2 since you concatenate the features (which creates twice as many).

    def forward(self, x, encoder_features):
        """Performs the forward pass for all blocks in the decoder.

        Parameters
        ----------
        x : torch.Tensor
            input to the decoder
        encoder_features: list
            contains the encoder features to be concatenated to the corresponding level of the decoder

        Returns
        -------
        x : torch.Tensor
            output of the decoder
        """
        for i in range(len(self.chs) - 1):
            # transposed convolution
            x = self.upconvs[i](x)
            # get the features from the corresponding level of the encoder
            enc_ftrs = encoder_features[i]
            # concatenate these features to x
            x = torch.cat([x, enc_ftrs], dim=1)
            # convolutional block
            x = self.dec_blocks[i](x)

        return x


class UNet(nn.Module):
    """A representation for a unet

    Parameters
    ----------
    enc_chs : tuple
        holds the number of input channels of each block in the encoder
    dec_chs : tuple
        holds the number of input channels of each block in the decoder
    num_classes : int
        number of output classes of the segmentation
    """

    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
        num_classes=1,
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], num_classes, 1),
        )  # output layer

    def forward(self, x):
        """Performs the forward pass of the unet.

        Parameters
        ----------
        x : torch.Tensor
            the input to the unet (image)

        Returns
        -------
        out : torch.Tensor
            unet output, the logits of the predicted segmentation mask
        """

        enc_ftrs = self.encoder(x)
        # the encoder features are the input to the decoder
        # we reverse the features first
        reverse_enc_ftrs = enc_ftrs[::-1]
        # this is because the lower level encoder features are
        # concatenated first
        # the last output of the encoder (0 index after reverse)
        # is the input to the decoder
        out = self.decoder(reverse_enc_ftrs[0], reverse_enc_ftrs[1:])
        # last layer ensure output has appropriate number of channels (1)
        out = self.head(out)
        return out