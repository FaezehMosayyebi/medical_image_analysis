##############################################################################################
#                                                                                            #
#     coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: This data loader allows you to customize the batch and patch sizes and    #
#     control the percentage of foreground classes in each patch. With this reliable batch    #
#     generator, you can be assured that every batch will be unique in each epoch.            #
#     Purpose: The primary purpose of this data loader is to efficiently handle data loading,#
#     batch generation, and patching while allowing you to specify the desired percentage of #
#     foreground classes.                                                                    #
#                                                                                            #
##############################################################################################

import numpy as np
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from builtins import object
import nibabel as nib


class DataLoader(object):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        __metaclass__ = ABCMeta
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.batchgenerator()

    @abstractmethod
    def batchgenerator(self):
        pass


class Batch_Loader(DataLoader):
    def __init__(
        self,
        sample_list,
        base_dir,
        patch_size,
        batch_size,
        oversample_foreground_percent=0.33,
        memmap_mode="r",
        pad_mode="edge",
        num_channels=1,
        num_classes=3,
    ):
        super(Batch_Loader, self).__init__(sample_list, batch_size, None)
        self.pad_kwargs_data = OrderedDict()
        self.sample_list = sample_list
        self.base_dir = base_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.oversample_foreground_percent = oversample_foreground_percent
        self.memmap_mode = memmap_mode
        self.pad_mode = pad_mode
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.list_container = [List for List in self.sample_list]

    def batchgenerator(self):
        need_to_pad = [0, 0, 0]

        # By the following algorithm we prevent our iterator to put same data in different batchs
        if len(self.sample_list) >= self.batch_size:
            selected_keys = np.random.choice(
                self.sample_list,
                size=self.batch_size,
                replace=False,  # Prevent to get duplicate data
            )

            for key in selected_keys:
                self.sample_list.remove(key)

        else:
            selected_keys = np.random.choice(
                self.sample_list,
                size=self.batch_size,
                replace=True,  # when the remained data is less than batch size we have to dunblicate data
            )

            for key in selected_keys:
                if key in self.sample_list:
                    self.sample_list.remove(key)

        if len(self.sample_list) == 0:
            self.sample_list = [sample for sample in self.list_container]

        # Defining a raw batch of image and label
        image = np.zeros(
            (self.batch_size, self.num_channels, *self.patch_size), dtype=np.float32
        )
        label = np.zeros((self.batch_size, *self.patch_size), dtype=np.float32)

        # Batching
        for i, ID in enumerate(selected_keys):
            if not i < round(
                self.batch_size * (1 - self.oversample_foreground_percent)
            ):  # We want to limit the voxels of a random channel in half the data of a batch
                force_fg = True
            else:
                force_fg = False

            # Loading images
            in_img = np.array(nib.load(self.base_dir + ID["image"]).get_fdata())
            in_seg = np.array(nib.load(self.base_dir + ID["label"]).get_fdata())

            # Patching

            # check if we need padding
            # Warning: you may need to change this part according to the dimention of your data
            for d in range(3):  # In hippocampuse data we have single channel 3D data
                if need_to_pad[d] + in_img.shape[d] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - in_img.shape[d]

            # Defining the set of voxels that the first voxel of a patch can be chosen from that
            # Warning: you may need to change this part according to the dimention of your data
            shape = in_img.shape
            lb_x = -need_to_pad[0] // 2
            ub_x = (
                shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            )
            lb_y = -need_to_pad[1] // 2
            ub_y = (
                shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            )
            lb_z = -need_to_pad[2] // 2
            ub_z = (
                shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]
            )

            if not force_fg:
                # choosing the first voxel of a patch randomly (ignoring the number of class voxels in a patch)
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            else:
                # Finding where the classes are and sampling some random locations ( max 10,000 samples per class)
                num_samples = 10000
                min_percent_coverage = (
                    0.01  # at least 1% of a random class voxels need to be selected.
                )
                rndst = np.random.RandomState(1234)
                class_locs = {}
                for c in range(1, self.num_classes):  # Ignore background channel
                    all_locs = np.argwhere(in_seg == c)
                    if len(all_locs) == 0:
                        class_locs[c] = []
                        continue
                    target_num_samples = min(num_samples, len(all_locs))
                    target_num_samples = max(
                        target_num_samples,
                        int(np.ceil(len(all_locs) * min_percent_coverage)),
                    )

                    selected = all_locs[
                        rndst.choice(len(all_locs), target_num_samples, replace=False)
                    ]
                    class_locs[c] = selected
                ######################################################################################################

                foreground_classes = np.array(
                    [i for i in class_locs if len(class_locs[i]) != 0]
                )

                # Warning: I think we don't need this because we have check befor to ignore background class
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    selected_class = None
                    voxels_of_that_class = None
                else:
                    selected_class = np.random.choice(foreground_classes)
                    voxels_of_that_class = class_locs[selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[
                        np.random.choice(len(voxels_of_that_class))
                    ]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            in_img = np.copy(
                in_img[
                    valid_bbox_x_lb:valid_bbox_x_ub,
                    valid_bbox_y_lb:valid_bbox_y_ub,
                    valid_bbox_z_lb:valid_bbox_z_ub,
                ]
            )

            # Warning: In case of hippocampuse that we don't have any channels in image dimention
            image[i, 0] = np.pad(
                in_img,
                (
                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)),
                ),
                self.pad_mode,
                **self.pad_kwargs_data
            )

            in_seg = np.copy(
                in_seg[
                    valid_bbox_x_lb:valid_bbox_x_ub,
                    valid_bbox_y_lb:valid_bbox_y_ub,
                    valid_bbox_z_lb:valid_bbox_z_ub,
                ]
            )

            label[i] = np.pad(
                in_seg,
                (
                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)),
                ),
                "constant",
                **{"constant_values": 0}
            )

        return {"data": image, "seg": label}
