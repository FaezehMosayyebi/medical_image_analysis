import torch
from torch.nn.functional import avg_pool3d
from augmentations import *


# SpatialTransform (Rotating and scaling)
class SpatialTransform(object):
    def __init__(
        self, p_rot_per_sample, rot_angle, p_scale_per_sample, scale_range, patch_size
    ):
        # Rotation configeration
        self.p_rot_per_sample = p_rot_per_sample
        self.rot_angle = rot_angle

        # scaling configeration
        self.p_scale_per_sample = p_scale_per_sample
        self.scale_range = scale_range

        self.patch_size = patch_size

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data_out, seg_out = augment_spatial(
            data,
            seg,
            self.patch_size,
            self.p_rot_per_sample,
            self.rot_angle,
            self.p_scale_per_sample,
            self.scale_range,
        )
        data_dict["data"] = data_out
        data_dict["seg"] = seg_out

        return data_dict


# GaussianNoiseTransform
class GaussianNoiseTransform(object):
    def __init__(self, noise_variance, p_per_sample, data_key="data"):
        self.noise_variance = noise_variance
        self.p_per_sample = p_per_sample
        self.data_key = data_key

    def __call__(self, **data_dict):
        for b in range(data_dict["data"].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data_dict["data"][b] = augment_gaussian_noise(
                    data_dict["data"][b], self.noise_variance
                )
        return data_dict


# GaussianBlurTransform
class GaussianBlurTransform(object):
    def __init__(self, blur_sigma, p_per_channel, p_per_sample):
        self.blur_sigma = blur_sigma
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        for b in range(data_dict["data"].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data_dict["data"][b] = augment_gaussian_blur(
                    data_dict["data"][b], self.blur_sigma, self.p_per_channel
                )
        return data_dict


# BrightnessMultiplicativeTransform
class BrightnessMultiplicativeTransform(object):
    def __init__(self, multiplier_range, p_per_sample):
        self.multiplier_range = multiplier_range
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(data_dict["data"].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data_dict["data"][b] = augment_brightness_multiplicative(
                    data_dict["data"][b], self.multiplier_range
                )
        return data_dict


# ContrastAugmentationTransform
class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range, p_per_sample):
        self.contrast_range = contrast_range
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(data_dict["data"].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data_dict["data"][b] = augment_contrast(
                    data_dict["data"][b], contrast_range=self.contrast_range
                )
        return data_dict


# SimulateLowResolutionTransform
class SimulateLowResolutionTransform(object):
    def __init__(
        self, zoom_range, p_per_channel, order_downsample, order_upsample, p_per_sample
    ):
        self.zoom_range = zoom_range  # Downsample Factor
        self.order_downsample = order_downsample
        self.order_upsample = order_upsample
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(data_dict["data"].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data_dict["data"][b] = augment_linear_downsampling_scipy(
                    data_dict["data"][b],
                    zoom_range=self.zoom_range,
                    p_per_channel=self.p_per_channel,
                    order_downsample=self.order_downsample,
                    order_upsample=self.order_upsample,
                )
        return data_dict


# GammaTransform
class GammaTransform(object):
    def __init__(self, gamma_range, p_per_sample):
        self.p_per_sample = p_per_sample = 0.15
        self.gamma_range = gamma_range

    def __call__(self, **data_dict):
        for b in range(data_dict["data"].shape[0]):
            # we perform gamma transformation on 15% of our data
            if np.random.uniform() < self.p_per_sample:
                data_dict["data"][b] = augment_gamma(
                    data_dict["data"][b], self.gamma_range
                )
        return data_dict


# MirrorTransform
class MirrorTransform(object):
    def __init__(self, axes, p_per_sample):
        self.p_per_sample = p_per_sample
        self.axes = axes

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                output_data, output_seg = augment_mirroring(
                    data[b], seg[b], axes=self.axes
                )
                data[b] = output_data
                seg[b] = output_seg

        data_dict["data"] = data
        data_dict["seg"] = seg

        return data_dict


# Downsample for deep supervision (only segmentation map)
def convert_seg_image_to_one_hot_encoding_batched(image, classes=None):
    if classes is None:
        classes = np.unique(image)

    output_shape = (
        [image.shape[0]] + [len(classes)] + list(image.shape[1:])
    )  # output shape = [b, c, x, y, z]
    out_image = np.zeros(output_shape, dtype=image.dtype)

    # one_hot coding
    for b in range(image.shape[0]):
        for i, c in enumerate(classes):
            out_image[b, i][image[b] == c] = 1

    return out_image


def downsample_seg_for_ds_transform(seg, ds_scales, classes):
    output = []
    one_hot = torch.from_numpy(
        convert_seg_image_to_one_hot_encoding_batched(seg, classes)
    )

    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(one_hot)
        else:
            kernel_size = tuple(int(1 / i) for i in s)
            stride = kernel_size
            pad = tuple((i - 1) // 2 for i in kernel_size)

            pool_op = avg_pool3d
            pooled = pool_op(
                one_hot,
                kernel_size,
                stride,
                pad,
                count_include_pad=False,
                ceil_mode=False,
            )

            output.append(pooled)
    return output


class DownsampleSegForDSTransform(object):
    def __init__(self, ds_scales, classes):
        self.classes = classes
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict["seg"] = downsample_seg_for_ds_transform(
            data_dict["seg"], self.ds_scales, self.classes
        )
        return data_dict


class NumpyToTensor(object):
    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, **data_dict):
        for key in self.keys:
            data_dict[key] = (
                torch.from_numpy(data_dict[key]).float().contiguous()
            )  # convert to tensor

        return data_dict
