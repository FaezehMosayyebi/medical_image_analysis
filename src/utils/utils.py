import numpy as np
import cv2
import imageio
from IPython.display import Image
import matplotlib.pyplot as plt
import pandas as pd


############################################# Generating GIF ##################################################
def get_labeled_image(image, label):
    image = image[0, :, :, :]  # getting flair
    classes = [0, 1, 2, 3]  # classes

    # one_hot coding
    out_seg = np.zeros(list(label.shape) + [len(classes)], dtype=label.dtype)
    for i, c in enumerate(classes):
        out_seg[:, :, :, i][label == c] = 1

    image = cv2.normalize(
        image[:, :, :],
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    ).astype(np.uint8)

    labeled_image = np.zeros_like(out_seg[:, :, :, 1:])

    # remove tumor part from image
    labeled_image[:, :, :, 0] = image * (out_seg[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (out_seg[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (out_seg[:, :, :, 0])

    # color labels
    labeled_image += out_seg[:, :, :, 1:] * 255

    return labeled_image


def visualize_data_gif(data_, output_dir):
    images = []
    print(data_.shape)
    for i in range(data_.shape[1]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave(output_dir + "/gif.gif", images, format="GIF", duration=0.01)
    return Image(filename=output_dir + "/gif.gif", format="PNG")


################################################# Making CSV File ################################################
class save_csv(object):
    def __init__(self, directory):
        self.directory = directory
        header = [
            "tr_loss",
            "val_loss",
            "tr_acc",
            "val_acc",
            "1_dice_s_tr",
            "2_dice_s_tr",
            "1_dice_s_val",
            "2_dice_s_val",
        ]
        self.df = pd.DataFrame(columns=header)

    def save(self, info):
        self.df.loc[info[0]] = info[1:]

    def end(self):
        self.df.to_csv(self.directory + "/model_history.csv")


def threedplot(threeDimage, slice_num, show_scale: bool, message: str):
    """
    Parameters:
                3Dimage(numpy array): input imagee
                slice nume (list pf positions [x, y, z]): The position of slice you want to see in each axes
                show_scale: If True you ca see axes scale
                message: The message you want as title of figure.
    Note: you have to import matplotlib.pyplot as plt
    """

    # plt.figure(figsize=(3,1))
    plt.suptitle(message, ha="left", va="bottom", fontsize=15)

    plt.subplot(131)
    plt.imshow(threeDimage[slice_num[0], :, :], "gray")
    plt.title("x")
    if not show_scale:
        plt.xticks([])
        plt.yticks([])

    plt.subplot(132)
    plt.imshow(threeDimage[:, slice_num[1], :], "gray")
    plt.title("y")

    if not show_scale:
        plt.xticks([])
        plt.yticks([])

    plt.subplot(133)
    plt.imshow(threeDimage[:, :, slice_num[2]], "gray")
    plt.title("z")
    if not show_scale:
        plt.xticks([])
        plt.yticks([])

    plt.show()
