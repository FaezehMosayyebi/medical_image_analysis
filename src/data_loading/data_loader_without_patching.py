##############################################################################################
#                                                                                            #
#     coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: A simple Dataloader and batch generator. With this reliable batch         #
#     generator, you can be assured that every batch will be unique in each epoch.           #
#     Purpose: Loading data and batch generating without patching data.                      #
#                                                                                            #
##############################################################################################

import numpy as np
from abc import ABCMeta, abstractmethod
from builtins import object
import nibabel as nib


class DataLoader(object):
    def __init__(self, data, batch_size):
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
        super(Batch_Loader, self).__init__(sample_list, batch_size)
        self.sample_list = sample_list
        self.base_dir = base_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.memmap_mode = memmap_mode
        self.num_channels = num_channels
        self.list_container = [List for List in self.sample_list]

    def batchgenerator(self):
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
            # Loading images
            in_img = np.array(nib.load(self.base_dir + ID["image"]).get_fdata())
            in_seg = np.array(nib.load(self.base_dir + ID["label"]).get_fdata())

            # Warning: In case of hippocampuse that we don't have any channels in image dimention
            image[i, 0] = in_img

            # Warning: If you have channeled images use the following code
            # image[i] = in_img

            label[i] = in_seg

        return {"data": image, "seg": label}
