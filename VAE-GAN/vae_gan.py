import torch
import torch.nn as nn
import architectures 

l1_loss = torch.nn.L1Loss()


class VAE_GAN(nn.Module):
    """A representation of the VAE-GAN

    Parameters
    ----------
    enc_chs : tuple 
        holds the number of input channels of each block in the encoder
    dec_chs : tuple 
        holds the number of input channels of each block in the decoder
    """
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
        dis_chs=(64,128,256,512)
        ):
        super().__init__()
        self.encoder = architectures.Encoder(chs=enc_chs)
        self.generator = architectures.Generator(chs=dec_chs)
        self.discriminator = architectures.Discriminator(chs=dis_chs)


    def forward(self, x):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.

        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder

        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)
        
        output = self.generator(latent_z)
        real_fake_value = self.discriminator(output)
        
        return real_fake_value, output, mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.
    
    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with 
    random numbers from the normal distribution.

    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted 
    latent distribution.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution

    Returns
    -------
    float
        the kld loss

    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_gan_loss(inputs, recons, mu, logvar):
    """Computes the VAE loss, sum of reconstruction and KLD loss

    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution

    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    return l1_loss(inputs, recons) + kld_loss(mu, logvar)
