##############################################################################################
#                                                                                            #
#     coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: The trainer class takes the database directory, output directory, name    #
#     of files as config, and the characteristic of the model to train it. The output of     #
#     this class is a model history in a CSV file, network progress figures, and a trained   #
#     network as a .pth.tar file.                                                            #
#     Purpose: Training the network.                                                         #
#                                                                                            #
##############################################################################################

# Imports
import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
import os
from time import time

#######################
from data_loading.data_loader import Batch_Loader
from data_augmentation.augmenter import data_augmentation
from network_architectures.CUNet_conjlayer_after_first_conv_with_bn import ComplexUNet
from Loss import DC_and_CE_loss, MultipleOutputLoss
from utils.utils import save_csv

#######################
import math


class trainer(object):
    def __init__(
        self,
        output_folder,
        dataset_directory,
        config,
        num_layers,
        model_type,
        conv_type,
    ):
        ################################ Hyperparameters##############################
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.batch_size = 9
        self.patch_size = (32, 32, 32)
        input_channels = 1
        base_num_features = 32
        self.model_type = model_type
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.epoch = 0

        self.load_model = False

        self.data_dir = dataset_directory
        self.output_dir = output_folder
        self.config = config

        ############# Warning num batch size starts from "0"
        self.num_batches_per_epoch = math.ceil(
            len(self.config["training"]) / self.batch_size
        )
        self.num_val_batches_per_epoch = math.ceil(
            len(self.config["validation"]) / self.batch_size
        )
        # self.num_batches_per_epoch = 5
        # self.num_val_batches_per_epoch = 1

        self.online_eval_tp_train = []
        self.online_eval_fp_train = []
        self.online_eval_fn_train = []
        self.online_eval_tp_valid = []
        self.online_eval_fp_valid = []
        self.online_eval_fn_valid = []

        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_tr_acc = []
        self.all_val_acc = []

        self.info = []

        self.ds_loss_weights = None

        self.batch_dice = True
        self.loss = DC_and_CE_loss({"batch_dice": self.batch_dice, "smooth": 1e-5}, {})

        # Here we wrap the loss for deep supervision
        # we need to know the number of outputs of the network

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2**i) for i in range(self.num_layers)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array(
            [True]
            + [
                True if i < self.num_layers - 1 else False
                for i in range(1, self.num_layers)
            ]
        )
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights
        # now wrap the loss
        self.loss = MultipleOutputLoss(self.loss, self.ds_loss_weights)

        ############################## Data #######################################
        self.dl_tr = Batch_Loader(
            self.config["training"],
            self.data_dir,
            self.patch_size,
            self.batch_size,
            0.33,
            "r",
            "edge",
            1,
            3,
        )
        self.dl_val = Batch_Loader(
            self.config["validation"],
            self.data_dir,
            self.patch_size,
            self.batch_size,
            0.33,
            "r",
            "edge",
            1,
            3,
        )

        self.tr_gen, self.val_gen = data_augmentation(
            self.dl_tr, self.dl_val, self.num_layers, self.patch_size
        )

        ############################# device ######################################
        if torch.cuda.is_available():
            print("cuda is available")
        else:
            print("the process will take palce in CPU")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ################## Network and optimizer initialization ####################
        self.network = ComplexUNet(
            input_channels,
            base_num_features,
            self.num_layers,
            do_ds=True,
            mode=self.model_type,
            conv_type=self.conv_type,
        )
        if self.load_model:
            # Loade Model
            checkpoint = torch.load("/content/output/My_checkpoint1-50.pth.tar")
            self.network.load_state_dict(checkpoint["state_dict"])

        self.network = self.network.to(device=self.device)

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )

    ##################################################################################
    def plot_progress(self):
        font = {"weight": "normal", "size": 18}

        matplotlib.rc("font", **font)

        fig_loss = plt.figure(figsize=(30, 24))
        fig_acc = plt.figure(figsize=(30, 24))

        # we have 1x1 window and select the first one
        ax_loss = fig_loss.add_subplot(111)
        ax_acc = fig_acc.add_subplot(111)

        x_values = list(range(self.epoch + 1))

        ax_loss.plot(
            x_values, self.all_tr_losses, color="b", ls="-", label="Train_Loss"
        )
        ax_loss.plot(
            x_values, self.all_val_losses, color="r", ls="-", label="Validation_Loss"
        )

        ax_acc.plot(
            x_values, self.all_tr_acc, color="b", ls="-", label="Train_Accuracy"
        )
        ax_acc.plot(
            x_values, self.all_val_acc, color="r", ls="-", label="Validation_Accuracy"
        )

        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")

        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")

        ax_loss.legend()
        ax_acc.legend()

        fig_loss.savefig(os.path.join(self.output_dir, "loss_progress.png"))
        plt.close()

        fig_acc.savefig(os.path.join(self.output_dir, "acc_progress.png"))
        plt.close()

    ################################################################################
    def run_online_evaluation(
        self, output_old, target_old, mode
    ):  # for calculating dice score in each epoch you can do this in traning and validation process
        # as we use deep supervision we have different resolutions. in this step we need full-resolution images
        target = target_old[0]
        output = output_old[0]

        with torch.no_grad():  # we want to prevent gradient propagation in network in this step
            num_classes = output.shape[1]
            output_seg = output.argmax(1)  # output_seg.shape=(b,x,y,z)
            target = target.argmax(1)

            tp_hard = torch.zeros((num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((num_classes - 1)).to(output_seg.device.index)

            for c in range(1, num_classes):  # ignores backgound
                tp_hard[c - 1] = (
                    (output_seg == c).float() * (target == c).float()
                ).sum()  # True-Positive
                fp_hard[c - 1] = (
                    (output_seg == c).float() * (target != c).float()
                ).sum()  # False_Positive
                fn_hard[c - 1] = (
                    (output_seg != c).float() * (target == c).float()
                ).sum()  # False_Negative
            tp_hard = (
                tp_hard.detach().cpu().numpy()
            )  # canceling gradiatn/ copy from cuda to system memory/convert to numpy array
            fp_hard = fp_hard.detach().cpu().numpy()
            fn_hard = fn_hard.detach().cpu().numpy()

            if mode == "train":
                self.online_eval_tp_train.append(list(tp_hard))
                self.online_eval_fp_train.append(list(fp_hard))
                self.online_eval_fn_train.append(list(fn_hard))
            elif mode == "validation":
                self.online_eval_tp_valid.append(list(tp_hard))
                self.online_eval_fp_valid.append(list(fp_hard))
                self.online_eval_fn_valid.append(list(fn_hard))
            else:
                print("Incorrect Mode")

    def finish_online_evaluation(self):
        self.online_eval_tp_train = np.sum(self.online_eval_tp_train, 0)
        self.online_eval_fp_train = np.sum(self.online_eval_fp_train, 0)
        self.online_eval_fn_train = np.sum(self.online_eval_fn_train, 0)
        self.online_eval_tp_valid = np.sum(self.online_eval_tp_valid, 0)
        self.online_eval_fp_valid = np.sum(self.online_eval_fp_valid, 0)
        self.online_eval_fn_valid = np.sum(self.online_eval_fn_valid, 0)

        global_dc_per_class_train = [
            i
            for i in [
                2 * i / (2 * i + j + k)
                for i, j, k in zip(
                    self.online_eval_tp_train,
                    self.online_eval_fp_train,
                    self.online_eval_fn_train,
                )
            ]
            if not np.isnan(i)
        ]
        global_dc_per_class_valid = [
            i
            for i in [
                2 * i / (2 * i + j + k)
                for i, j, k in zip(
                    self.online_eval_tp_valid,
                    self.online_eval_fp_valid,
                    self.online_eval_fn_valid,
                )
            ]
            if not np.isnan(i)
        ]

        self.all_tr_acc.append(np.mean(global_dc_per_class_train))
        self.all_val_acc.append(np.mean(global_dc_per_class_valid))

        self.info.append(self.all_tr_acc[-1])
        self.info.append(self.all_val_acc[-1])

        for tr_dice in global_dc_per_class_train:
            self.info.append(tr_dice)

        for val_dice in global_dc_per_class_valid:
            self.info.append(val_dice)

        print(
            "Average global foreground Dice-Training:",
            [np.round(i, 4) for i in global_dc_per_class_train],
        )
        print(
            "Average global foreground Dice-Validation:",
            [np.round(i, 4) for i in global_dc_per_class_valid],
        )
        print("Train Accuracy: %.4f" % self.all_tr_acc[-1])
        print("validation Accuracy: %.4f" % self.all_val_acc[-1])

        self.online_eval_tp_train = []
        self.online_eval_fp_train = []
        self.online_eval_fn_train = []
        self.online_eval_tp_valid = []
        self.online_eval_fp_valid = []
        self.online_eval_fn_valid = []

    ################################################################################
    def run_iteration(self, data_generator, do_backprop=True, mode="train"):
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["seg"]

        scaler = GradScaler()

        data = data.to(device=self.device)
        target = [i.to(device=self.device) for i in target]

        # forward
        self.optimizer.zero_grad()  # making zero all the gradients from previouse iterations

        # using fp16: for reducing vram and speeding up training
        with torch.cuda.amp.autocast():
            output = self.network(data)  # orediction
            del data  # deletes data to gain more ram because we don't need it anymore

            l = self.loss(output, target)  # computing loss

        # backward
        if do_backprop:  # this condition help us to choose train and valid mode
            scaler.scale(l).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            scaler.step(self.optimizer)
            scaler.update()

        self.run_online_evaluation(output, target, mode)

        del target

        return l.detach().cpu().numpy()

    ################################################################################
    def run_training(self):
        # _ = self.tr_gen.next()
        # _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        csv = save_csv(self.output_dir)

        # running training
        while self.epoch < self.max_num_epochs:
            print("\nepoch: ", self.epoch + 1)  # epoch start from 0
            self.info.append(self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()  # we put the network at train mode

            # progress bar
            with trange(
                self.num_batches_per_epoch
            ) as tbar:  # Waning! set the right num of batches
                for b in tbar:
                    tbar.set_description(
                        "Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs)
                    )

                    l = self.run_iteration(self.tr_gen, True, "train")

                    tbar.set_postfix(loss=l)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))

            self.info.append(self.all_tr_losses[-1])
            print("Train Loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(
                    self.num_val_batches_per_epoch
                ):  # Warning! set the write num of batches
                    l = self.run_iteration(self.val_gen, False, "validation")
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))

                self.info.append(self.all_val_losses[-1])
                print("Validation Loss: %.4f" % self.all_val_losses[-1])

            self.optimizer.param_groups[0]["lr"] = (
                self.initial_lr * (1 - self.epoch / self.max_num_epochs) ** 0.9
            )  # poly-rate learning rate
            self.finish_online_evaluation()

            # we can also add early stopping

            epoch_end_time = time()

            csv.save(self.info)
            self.info = []
            self.plot_progress()
            self.epoch += 1
            print("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.
        csv.end()

        # save checkpoints
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.output_dir + "/My_checkpoint.pth.tar")
