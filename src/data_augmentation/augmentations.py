import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


# Spatial augment
def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = np.dot(
        rot_matrix,
        np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)],
            ]
        ),
    )
    rot_matrix = np.dot(
        rot_matrix,
        np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_y), -np.sin(angle_y)],
                [0, np.sin(angle_y), np.cos(angle_y)],
            ]
        ),
    )
    rot_matrix = np.dot(
        rot_matrix,
        np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_z), -np.sin(angle_z)],
                [0, np.sin(angle_z), np.cos(angle_z)],
            ]
        ),
    )
    coords = (
        np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix)
        .transpose()
        .reshape(coords.shape)
    )
    return coords


def augment_spatial(
    data, seg, patch_size, p_rot_per_sample, rot_angle, p_scale_per_sample, scale_range
):
    seg_result = np.zeros((seg.shape[0], *patch_size), dtype=np.float32)
    data_result = np.zeros(
        (data.shape[0], data.shape[1], *patch_size),
        dtype=np.float32,
    )

    for sample_id in range(data.shape[0]):  # batch size
        # calculating coords
        tmp = tuple([np.arange(i) for i in patch_size])
        coords = np.array(np.meshgrid(*tmp, indexing="ij")).astype(float)
        for d in range(len(patch_size)):
            coords[d] -= ((np.array(patch_size).astype(float) - 1) / 2.0)[d]

        # Rotation
        # we rotate 20% of data
        if np.random.uniform() < p_rot_per_sample:
            a_x = np.random.uniform(rot_angle[0], rot_angle[1])
            a_y = np.random.uniform(rot_angle[0], rot_angle[1])
            a_z = np.random.uniform(rot_angle[0], rot_angle[1])
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)

        # Scaling
        # we scale 20% of data
        if np.random.uniform() < p_scale_per_sample:
            # Zoom in 50% of data and zoom out the other 50%
            if np.random.random() < 0.5:
                scale = np.random.uniform(scale_range[0], 1)
            else:
                scale = np.random.uniform(max(scale_range[0], 1), scale_range[1])

            coords *= scale

        # now find a nice center location
        for d in range(3):
            ctr = data.shape[d + 2] / 2.0 - 0.5
            coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = map_coordinates(
                    data[sample_id, channel_id].astype(float),
                    coords,
                    order=3,
                    mode="nearest",
                    cval=0,
                ).astype(data[sample_id, channel_id].dtype)
                seg_result[sample_id] = map_coordinates(
                    seg[sample_id].astype(float),
                    coords,
                    order=0,
                    mode="constant",
                    cval=0,
                ).astype(seg[sample_id].dtype)
    return data_result, seg_result


# GaussianNoise
def augment_gaussian_noise(data_sample, noise_variance):
    # finding a random variance in variance range
    variance = np.random.uniform(noise_variance[0], noise_variance[1])

    output = np.zeros(data_sample.shape)

    # adding noise to all channels
    for c in range(output.shape[0]):
        output[c] = data_sample[c] + np.random.normal(
            0.0, variance, size=data_sample[c].shape
        )  # Adding noise
    return output


# GaussianBlur
def augment_gaussian_blur(
    data_sample: np.ndarray, sigma_range, p_per_channel
) -> np.ndarray:
    # Applyrin the process to all channels
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:  # choosing channels randomly
            sigma = np.random.uniform(
                sigma_range[0], sigma_range[1]
            )  # finding a random sigma
            data_sample[c] = gaussian_filter(
                data_sample[c], sigma, order=0
            )  # gaussian filter
    return data_sample


# BrightnessMultiplicative
def augment_brightness_multiplicative(data_sample, multiplier_range):
    output = np.zeros(data_sample.shape)

    for c in range(output.shape[0]):  # applying the process to all the channels
        multiplier = np.random.uniform(
            multiplier_range[0], multiplier_range[1]
        )  # chossing random multiplier
        output[c] = data_sample[c] * multiplier  # multiplying
    return output


# ContrastAugmentation
def augment_contrast(data_sample, contrast_range) -> np.ndarray:
    output = np.zeros(data_sample.shape)

    for c in range(output.shape[0]):
        if np.random.random() < 0.5:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

        mn = data_sample[c].mean()  # mean
        minm = data_sample[c].min()  # min
        maxm = data_sample[c].max()  # max

        output[c] = (data_sample[c] - mn) * factor + mn  # changing contrast
        output[c][output[c] < minm] = minm  # limiting voxels intensities
        output[c][output[c] > maxm] = maxm  # ~~~~~~~~~~~~~~~

    return output


# SimulateLowResolution
def augment_linear_downsampling_scipy(
    data_sample, zoom_range, p_per_channel, order_downsample, order_upsample
):
    output = np.zeros(data_sample.shape)

    orig_shape = np.array(data_sample.shape[1:])

    for c in range(output.shape[0]):
        if np.random.uniform() < p_per_channel:
            zoom = np.random.uniform(zoom_range[0], zoom_range[1])
            target_shape = np.round(orig_shape * zoom).astype(
                int
            )  # calculating new shape
            downsampled = resize(
                data_sample[c].astype(float),
                target_shape,
                order=order_downsample,
                mode="edge",
                anti_aliasing=False,
            )  # resize the image to new shape
            output[c] = resize(
                downsampled,
                orig_shape,
                order=order_upsample,
                mode="edge",
                anti_aliasing=False,
            )  # resizing back to original shape (it causes lower resolution)

    return data_sample


# Gamma
def augment_gamma(data_sample, gamma_range, epsilon=1e-7):
    # inverting Image
    data_sample = -data_sample

    # doing all processes on per channel
    for c in range(data_sample.shape[0]):
        # finding mean and Standard deviation to scale back voxel intensities to their original factor
        mn = data_sample[c].mean()
        sd = data_sample[c].std()

        # transforms half of data with the gamma range of [0.7, 1] and the other half with the gamma range of [1, 1.5]
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])

        # finding min and max to scale pach intensities to a factor of [0, 1]
        minm = data_sample[c].min()
        rnge = data_sample[c].max() - minm

        # Scaling to a factor of [0, 1] as well as gamma transforming
        data_sample[c] = (
            np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma)
            * float(rnge + epsilon)
            + minm
        )

        # Scaling back voxel intensities to their original value range
        data_sample[c] = data_sample[c] - data_sample[c].mean()
        data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
        data_sample[c] = data_sample[c] + mn

    # mirroring back
    data_sample = -data_sample
    return data_sample


# Mirror
def augment_mirroring(sample_data, sample_seg, axes=(2, 3, 4)):
    out_data = np.zeros(sample_data.shape)
    out_seg = np.zeros(sample_seg.shape)

    # mirroring over X axes
    if 2 in axes and np.random.uniform() < 0.5:
        out_data[:, :] = sample_data[:, ::-1]
        out_seg[:] = sample_seg[::-1]

    # mirroring over Y axes
    if 3 in axes and np.random.uniform() < 0.5:
        out_data[:, :, :] = sample_data[:, :, ::-1]
        out_seg[:, :] = sample_seg[:, ::-1]

    # mirroring over Z axes
    if 4 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            out_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            out_seg[:, :, :] = sample_seg[:, :, ::-1]
    return out_data, out_seg
