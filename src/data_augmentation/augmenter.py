from transformers import *


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict


class Augmenter(object):
    def __init__(self, data_loader, transform):
        self.data_loader = data_loader
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.data_loader)
        if self.transform is not None:
            item = self.transform(**item)
        return item

    def next(self):
        return self.__next__()


def data_augmentation(dataloader_train, dataloader_valid, num_layers, patch_size):
    net_num_pool_op_kernel_sizes = []
    for i in range(num_layers):
        net_num_pool_op_kernel_sizes.append([2, 2, 2])

    deep_supervision_scales = [[1, 1, 1]] + list(
        list(i) for i in 1 / np.cumprod(np.vstack(net_num_pool_op_kernel_sizes), axis=0)
    )[:-1]
    classes = [0, 1, 2]

    tr_transforms = []

    tr_transforms.append(
        SpatialTransform(
            p_rot_per_sample=0.2,
            rot_angle=((-30.0 / 360) * (2.0 * np.pi), (30.0 / 360) * (2.0 * np.pi)),
            p_scale_per_sample=0.2,
            scale_range=(0.7, 1.4),
            patch_size=patch_size,
        )
    )
    tr_transforms.append(
        GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=0.15)
    )
    tr_transforms.append(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.5), p_per_channel=0.5, p_per_sample=0.2
        )
    )
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.7, 1.3), p_per_sample=0.15
        )
    )
    tr_transforms.append(
        ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15)
    )
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(1, 2),
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
        )
    )
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), p_per_sample=0.15))
    tr_transforms.append(MirrorTransform(axes=(2, 3, 4), p_per_sample=0.5))
    tr_transforms.append(
        DownsampleSegForDSTransform(deep_supervision_scales, classes=classes)
    )
    tr_transforms.append(NumpyToTensor(["data"]))  # seg is already tensor
    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = Augmenter(dataloader_train, tr_transforms)

    val_transforms = []
    val_transforms.append(
        DownsampleSegForDSTransform(deep_supervision_scales, classes=classes)
    )
    val_transforms.append(NumpyToTensor(["data"]))  # seg is already tensor
    val_transforms = Compose(val_transforms)

    batchgenerator_valid = Augmenter(dataloader_valid, val_transforms)

    return batchgenerator_train, batchgenerator_valid
