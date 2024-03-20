import torch
import torch.nn as nn
import re
import torch.nn.functional as F


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
#
# Parameters
# -----------------------
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |out_ch|: the #channels of the normalized activations, hence the output dim of SPADE
# |in_ch_seg|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, in_ch_seg, out_ch):
        super().__init__()


        assert config_text.startswith('spade')
        parsed = re.search(r'spade(\D+)(\d)x(\d)', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(out_ch, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = nn.SynchronizedBatchNorm2d(out_ch, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(out_ch, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded. 
        # (Bram: inherited from SPADE we can play with this value)
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_ch_seg, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, out_ch, kernel_size=ks, padding=pw) # TODO SPADE feature maps nr defined as same as layer
        self.mlp_beta = nn.Conv2d(nhidden, out_ch, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap.float(), size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out
    
class Block(nn.Module):
    """Basic convolutional building block

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu =  nn.ReLU() # normal reLU
        self.bn1 = nn.BatchNorm2d(num_features=out_ch) 
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)

    def forward(self, x):
        """Performs a forward pass of the block
       
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        # use batch normalisation
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)        

        return x

class LeakyBlock(nn.Module):
    """Basic convolutional building block with leaky reLU

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.relu =  nn.LeakyReLU()  # leaky ReLU
        self.bn1 = nn.BatchNorm2d(num_features=out_ch) # batch normalisation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)  

    def forward(self, x):
        """Performs a forward pass of the block
       
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        # use batch normalisation
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)        

        # TODO
        return x

class SPADEBlock(nn.Module):
    """Convolutional building block with SPADE implementation

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    in_ch_seg : int
        number of channels of the input image default=1
    """

    def __init__(self, in_ch, out_ch, in_ch_seg=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu =  nn.LeakyReLU() 
        self.spade = SPADE(config_text='spadebatch3x3',in_ch_seg=in_ch_seg,out_ch=out_ch) 
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.spade2 = SPADE(config_text='spadebatch3x3',in_ch_seg=in_ch_seg,out_ch=out_ch) 

    def forward(self, x, labels):
        """Performs a forward pass of the block
       
        x : torch.Tensor
            the input to the block
        labels : torch.Tensor
            the segmentation map or labels to the corresponding image
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        # use batch normalisation
        x = self.conv1(x)
        x = self.relu(x)
        x = self.spade(x, labels)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.spade2(x, labels)        

        return x