import torch
import torch.nn as nn
import blocks




class Encoder(nn.Module):
    """The encoder part of the VAE-GAN

    Parameters
    ----------
    spatial_size : list[int]
        size of the input image, by default [64, 64]
    z_dim : int
        dimension of the latent space
    chs : tuple
        hold the number of input channels for each encoder block
    """

    def __init__(self, spatial_size=[64, 64], z_dim=256, chs=(1, 64, 128, 256)):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [blocks.LeakyBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(2)
        # height and width of images at lowest resolution level
        _h, _w = int(spatial_size[0]/(2**(len(chs)-1))), int(spatial_size[1]/(2**(len(chs)-1)))

        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input image to the encoder

        Returns
        -------
        list[torch.Tensor]    
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """

        for block in self.enc_blocks:
            # TODO: conv block     
            x = block(x)      
            # TODO: pooling
            x = self.pool(x) 
        # TODO: output layer
        x = self.out(x)          
        return torch.chunk(x, 2, dim=1)  # 2 chunks, 1 each for mu and logvar


class Generator(nn.Module):
    """Generator of the VAE-GAN

    Parameters
    ----------
    z_dim : int 
        dimension of latent space
    chs : tuple
        holds the number of channels for each block
    h : int, optional
        height of image at lowest resolution level, by default 8
    w : int, optional
        width of image at lowest resolution level, by default 8    
    """

    def __init__(self, labels=None, z_dim=256, chs=(256, 128, 64, 32), h=8, w=8):

        super().__init__()
        self.chs = chs
        self.h = h  
        self.w = w  
        self.z_dim = z_dim  
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping

        self.upconvs = nn.ModuleList(
            # TODO: transposed convolution    
            [nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs) - 1)]                       
        )

        self.dec_blocks = nn.ModuleList(
            [blocks.SPADEBlock(chs[i], chs[i+1]) for i in range(len(chs)- 1)]     #note the use of spade blocks      
        )
        self.head = nn.Conv2d(chs[-1], 1, kernel_size=3, padding=1)

    def forward(self, z, labels):
        """Performs the forward pass of decoder

        Parameters
        ----------
        z : torch.Tensor
            input to the generator
        
        Returns
        -------
        x : torch.Tensor
        
        """
        x = self.proj_z(z)#  fully connected layer
        x = self.reshape(x) #  reshape to image dimensions
        for i in range(len(self.chs) - 1):
            # transposed convolution
            x = self.upconvs[i](x)
            # convolutional SPADE block
            x = self.dec_blocks[i](x, labels)
        x = self.head(x)
        return x
    

class Discriminator(nn.Module):
    """ Discriminator of the VAE-GAN
    inspired on the discriminator from the SPADE paper.
    
    The block are defined here and not in blocks.py because of their highly specific nature.
    """
    def __init__(self, chs=(64,128,256,512)):
        super().__init__()

        sequence = [[nn.Conv2d(1, chs[0], kernel_size=3, stride=2, padding=1), 
                     nn.LeakyReLU()]]
        
        for i in range(0,len(chs)-1):
            stride = 1 if i == len(chs)-2 else 2
            sequence += [[nn.Sequential(
                            nn.Conv2d(chs[i], chs[i+1], kernel_size=3, stride=stride, padding=1),
                            nn.BatchNorm2d(num_features=chs[i+1])
                            ),
                          nn.LeakyReLU()]]

        self.dis_blocks = []
        for n in range(len(sequence)):
            self.dis_blocks.append(nn.Sequential(*sequence[n]))
        self.dis_blocks = nn.ModuleList(self.dis_blocks)
        self.out = nn.Conv2d(chs[-1], 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        for block in self.dis_blocks:
            x = block(x)
        x = self.out(x)

        return x
        