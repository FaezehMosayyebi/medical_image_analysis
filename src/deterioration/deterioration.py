##############################################################################################
#                                                                                            #
#     coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: These codes are a Python implementation of the algorithm published in     #
#                   https://arxiv.org/abs/2208.14360 based on                                #
#                   https://github.com/Mostafa-Ghazi/MRI-Augmentation                        #
#     Purpose: I aim to use these codes to make deteriorated data to check the ability       #
#              of my network to deal with deteriorated data.                                 #
#                                                                                            #
##############################################################################################


import numpy as np
import math


# bias field
def bias(image):
    grid = np.mgrid[0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]]
    x0 = np.random.uniform(0, image.shape[0])
    y0 = np.random.uniform(0, image.shape[1])
    z0 = np.random.uniform(0, image.shape[2])
    G = 1 - (
        ((grid[0] - x0) ** 2) / (image.shape[0] ** 2)
        + ((grid[1] - y0) ** 2) / (image.shape[1] ** 2)
        + ((grid[2] - z0) ** 2) / (image.shape[2] ** 2)
    )
    bias_feild_image = G * image
    return bias_feild_image


# Gibbs Ringing
def GibbsRinging(imageVolume, numSample, p):
    imageHeight, imageWidth, imageDepth = imageVolume.shape  # volume size

    if p == 1:  # along x-axis
        imageVolume = np.transpose(imageVolume, (0, 2, 1))  # [Height, Depth, Width]
        imageVolume = np.fft.fftshift(
            np.fft.fftn(imageVolume, (imageHeight, imageDepth, imageWidth))
        )  # centralized fast Fourier transform (k-space)
        imageVolume[:, :, 0 : math.ceil(imageWidth / 2) - math.ceil(numSample / 2)] = 0
        imageVolume[
            :, :, math.ceil(imageWidth / 2) + math.ceil(numSample / 2) : imageWidth
        ] = 0  # truncated high frequency components
        imageVolume = np.abs(
            np.fft.ifftn(
                np.fft.ifftshift(imageVolume), (imageHeight, imageDepth, imageWidth)
            )
        )  # inverse fft of decentralized truncated data
        imageVolume = np.transpose(imageVolume, (0, 2, 1))  # [Height, Width, Depth]

    elif p == 2:  # along y-axis
        imageVolume = np.transpose(imageVolume, (1, 2, 0))  # [Width, Depth, Height]
        imageVolume = np.fft.fftshift(
            np.fft.fftn(imageVolume, (imageWidth, imageDepth, imageHeight))
        )
        imageVolume[:, :, 0 : math.ceil(imageHeight / 2) - math.ceil(numSample / 2)] = 0
        imageVolume[
            :, :, math.ceil(imageHeight / 2) + math.ceil(numSample / 2) : imageHeight
        ] = 0
        imageVolume = np.abs(
            np.fft.ifftn(
                np.fft.ifftshift(imageVolume), (imageWidth, imageDepth, imageHeight)
            )
        )
        imageVolume = np.transpose(imageVolume, (2, 0, 1))  # [Height, Width, Depth]

    elif p == 3:  # along z-axis
        imageVolume = np.fft.fftshift(
            np.fft.fftn(imageVolume, (imageHeight, imageWidth, imageDepth))
        )
        imageVolume[:, :, 0 : math.ceil(imageDepth / 2) - math.ceil(numSample / 2)] = 0
        imageVolume[
            :, :, math.ceil(imageDepth / 2) + math.ceil(numSample / 2) : imageDepth
        ] = 0
        imageVolume = np.abs(
            np.fft.ifftn(
                np.fft.ifftshift(imageVolume), (imageHeight, imageWidth, imageDepth)
            )
        )

    return imageVolume


# Motion Ghosting
def motionGhosting(imageVolume, alpha, numReps, p):
    imageHeight, imageWidth, imageDepth = imageVolume.shape
    imageVolume = np.fft.fftn(
        imageVolume, (imageHeight, imageWidth, imageDepth)
    )  # fast Fourier transform (k-space)

    if p == 1:  # along y-axis
        imageVolume[0:numReps:imageHeight, :, :] = (
            alpha * imageVolume[0:numReps:imageHeight, :, :]
        )  # k-space lines are modulated differently

    elif p == 2:  # along x-axis
        imageVolume[:, 0:numReps:imageWidth, :] = (
            alpha * imageVolume[:, 0:numReps:imageWidth, :]
        )  # k-space lines are modulated differently

    elif p == 3:  # along z-axis
        imageVolume[:, :, 0:numReps:imageDepth] = (
            alpha * imageVolume[:, :, 0:numReps:imageDepth]
        )  # k-space lines are modulated differently

    imageVolume = np.abs(
        np.fft.ifftn(imageVolume, (imageHeight, imageWidth, imageDepth))
    )  # inverse ifft of modified data
    return imageVolume


# Additive Noise
def gaussian_noise(imageVolume, noise_variance, mean=0.0):
    imageVolume = imageVolume + np.random.normal(
        mean, noise_variance, size=imageVolume.shape
    )  # Adding noise

    return imageVolume


# multiplicative Noise
def speckle_noise(imageVolume, noise_variance, mean=0.0):
    imageVolume = imageVolume + imageVolume * np.random.normal(
        mean, noise_variance, imageVolume.shape
    )

    return imageVolume


# Contrast change
def contrastChange(imageVolume, gamma):
    mn = imageVolume.mean()  # mean
    minm = imageVolume.min()  # min
    maxm = imageVolume.max()  # max

    imageVolume = (imageVolume - mn) * gamma + mn  # changing contrast
    imageVolume[imageVolume < minm] = minm  # limiting voxels intensities
    imageVolume[imageVolume > maxm] = maxm  # ~~~~~~~~~~~~~~~

    return imageVolume
