##############################################################################################
#                                                                                            #
#     coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: These codes are used to deteriorate data.                                 #
#     Purpose: I aim to use these codes to make deteriorated data to check the ability       #
#              of my network to deal with deteriorated data.                                 #
#                                                                                            #
##############################################################################################

import numpy as np
import nibabel as nib
import os
import json
import shutil
from deterioration import *

if __name__ == "__main__":
    with open(
        "/content/drive/MyDrive/hippocampus/hippocampus_json_files/fold0_hippocampus_dataset.json"
    ) as json_file:
        config = json.load(json_file)
    base_dir = "/content/dataset/content/cropped_test_hippocampus"
    out_dir = "/content/hippocampus_test_contrastchange_0.5"

    if not os.path.isdir(out_dir + "/imagesTr"):
        os.makedirs(out_dir + "/imagesTr")

    for ID in config["test"]:
        in_image = np.array(nib.load(base_dir + ID["image"]).get_fdata())
        # You can define your detrirations here
        out_image = contrastChange(in_image, 1.5)

        ni_img = nib.Nifti1Image(out_image, None)
        nib.save(ni_img, out_dir + ID["image"])

    shutil.copytree(base_dir + "/labelsTr", out_dir)
