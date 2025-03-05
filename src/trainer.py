import torch
import numpy as np
from torch.cuda.amp import GradScaler
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
import os
from time import time
import yaml
import json
from data_loading.data_loader import BatchLoader
from data_augmentation.augmenter import data_augmentation
from network import UNet
from Loss import DC_and_CE_loss, MultipleOutputLoss
from utils.utils import save_csv
from utils.log import info_logger


class Trainer(object):
    def __init__(self):
        super().__init__()
        with open("config.yml", "r") as config_file:
            self.config = yaml.safe_load(config_file)

        with open(self.config["data_config"]["sample_list_file_path"]) as file:
            sample_list = json.load(file)
        batch_size = self.config["data_config"]["batch_size"]
        patch_size = (
            self.config["data_config"]["patch_size"]["x"],
            self.config["data_config"]["patch_size"]["y"],
            self.config["data_config"]["patch_size"]["z"],
        )

        num_layers = self.config["model_config"]["unet_num_layers"]
        input_channels = self.config["model_config"]["num_input_channels"]

        self.num_classes = self.config["data_config"]["num_classes"]

        self.max_num_epochs = self.config["train_config"]["num_epochs"]
        self.initial_lr = self.config["train_config"]["initial_learning_rate"]
        self.epoch = 0

        self.output_dir = self.config["output_folder"]

        self.csv = save_csv(self.output_dir)
        self.info = []

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

        self.loss = DC_and_CE_loss(
            {"batch_dice": self.config["loss_config"]["batch_dice"], "smooth": 1e-5}, {}
        )
        self.best_loss = float("inf")

        # Here we wrap the loss for deep supervision
        # we need to know the number of outputs of the network

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        ds_loss_weights = self.config["loss_config"]["ds_loss_weights"]
        if ds_loss_weights is None:
            weights = np.array([1 / (2**i) for i in range(num_layers)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array(
                [True]
                + [True if i < num_layers - 1 else False for i in range(1, num_layers)]
            )
            weights[~mask] = 0
            weights = weights / weights.sum()
            ds_loss_weights = weights

        self.loss = MultipleOutputLoss(self.loss, ds_loss_weights)

        # Data
        self.dl_tr = BatchLoader(
            sample_list=sample_list["training"],
            base_dir=self.config["data_config"]["dataset_directory"],
            patch_size=patch_size,
            batch_size=batch_size,
            oversample_foreground_percent=self.config["data_config"][
                "oversample_foreground_percent"
            ],
            memmap_mode=self.config["data_config"]["memmap_mode"],
            pad_mode=self.config["data_config"]["pad_mode"],
            num_channels=input_channels,
            num_classes=self.num_classes,
        )
        self.dl_val = BatchLoader(
            sample_list=sample_list["validation"],
            base_dir=self.config["data_config"]["dataset_directory"],
            patch_size=patch_size,
            batch_size=batch_size,
            oversample_foreground_percent=self.config["data_config"][
                "oversample_foreground_percent"
            ],
            memmap_mode=self.config["data_config"]["memmap_mode"],
            pad_mode=self.config["data_config"]["pad_mode"],
            num_channels=input_channels,
            num_classes=self.num_classes,
        )

        self.tr_gen, self.val_gen = data_augmentation(
            self.dl_tr, self.dl_val, num_layers, patch_size, self.num_classes
        )

        # Device
        if torch.cuda.is_available():
            print("cuda is available")
        else:
            print("the process will take place in CPU")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network and Optimizer
        self.network = UNet(
            input_channels=input_channels,
            base_num_features=self.config["model_config"]["base_num_features"],
            num_layers=num_layers,
            do_ds=True,
        )

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.config["train_config"]["weight_decay"],
            momentum=self.config["train_config"]["optimizer_momentum"],
            nesterov=True,
        )

        if self.config["model_config"]["load_model"]:
            # Load Model
            checkpoint = torch.load(self.config["model_config"]["model_path"])
            self.network.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_loss = checkpoint["loss"]

        self.network = self.network.to(device=self.device)

    def _plot_progress(self):
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

    def _run_online_evaluation(self, output_old, target_old, validation: bool):
        """
        for calculating dice score in each epoch you can do this in training and validation process
        as we use deep supervision we have different resolutions. in this step we need full-resolution images
        """

        target = target_old[0]
        output = output_old[0]

        with torch.no_grad():
            num_classes = self.num_classes
            output_seg = output.argmax(1)  # output_seg.shape=(b,x,y,z)
            target = target.argmax(1)

            tp_hard = torch.zeros((num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((num_classes - 1)).to(output_seg.device.index)

            for c in range(1, num_classes):  # ignores background
                tp_hard[c - 1] = (
                    (output_seg == c).float() * (target == c).float()
                ).sum()
                fp_hard[c - 1] = (
                    (output_seg == c).float() * (target != c).float()
                ).sum()
                fn_hard[c - 1] = (
                    (output_seg != c).float() * (target == c).float()
                ).sum()

            tp_hard = tp_hard.detach().cpu().numpy()
            fp_hard = fp_hard.detach().cpu().numpy()
            fn_hard = fn_hard.detach().cpu().numpy()

            if not validation:
                self.online_eval_tp_train.append(list(tp_hard))
                self.online_eval_fp_train.append(list(fp_hard))
                self.online_eval_fn_train.append(list(fn_hard))
            else:
                self.online_eval_tp_valid.append(list(tp_hard))
                self.online_eval_fp_valid.append(list(fp_hard))
                self.online_eval_fn_valid.append(list(fn_hard))

    def _finish_online_evaluation(self):
        self.online_eval_tp_train = np.sum(self.online_eval_tp_train, 0)
        self.online_eval_fp_train = np.sum(self.online_eval_fp_train, 0)
        self.online_eval_fn_train = np.sum(self.online_eval_fn_train, 0)
        self.online_eval_tp_valid = np.sum(self.online_eval_tp_valid, 0)
        self.online_eval_fp_valid = np.sum(self.online_eval_fp_valid, 0)
        self.online_eval_fn_valid = np.sum(self.online_eval_fn_valid, 0)

        global_dc_per_class_train = [
            i
            for i in [
                2 * tp / (2 * tp + fp + fn)
                for tp, fp, fn in zip(
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
                2 * tp / (2 * tp + fp + fn)
                for tp, fp, fn in zip(
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

        print("Train Accuracy: %.4f" % self.all_tr_acc[-1])
        print("validation Accuracy: %.4f" % self.all_val_acc[-1])

        self.online_eval_tp_train = []
        self.online_eval_fp_train = []
        self.online_eval_fn_train = []
        self.online_eval_tp_valid = []
        self.online_eval_fp_valid = []
        self.online_eval_fn_valid = []

    def _run_iteration(self, data_generator, do_backprop):
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["seg"]

        data = data.to(device=self.device)
        target = [i.to(device=self.device) for i in target]

        scaler = GradScaler()

        # forward
        self.optimizer.zero_grad()
        # using fp16: for reducing V-RAM and speeding up training
        with torch.cuda.amp.autocast():
            output = self.network(data)
            del data
            loss = self.loss(output, target)

        # backward
        if do_backprop:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config["train_config"]["max_norm_gradient"],
            )
            scaler.step(self.optimizer)
            scaler.update()

        validation = not do_backprop
        self._run_online_evaluation(output, target, validation)

        del target

        return loss.detach().cpu().numpy()

    def run_training(self):
        # _ = self.tr_gen.next()
        # _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # running training
        while self.epoch < self.max_num_epochs:
            print("\nepoch: ", self.epoch + 1)
            self.info.append(self.epoch + 1)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            # progress bar
            with trange(self.config["train_config"]["num_batches_per_epoch"]) as tbar:
                for b in tbar:
                    tbar.set_description(
                        "Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs)
                    )

                    loss = self._run_iteration(self.tr_gen, True)

                    tbar.set_postfix(loss=loss)  # loss of each batch
                    train_losses_epoch.append(loss)

            self.all_tr_losses.append(np.mean(train_losses_epoch))  # loss of each epoch

            self.info.append(self.all_tr_losses[-1])
            print("Train Loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                self.network.eval()
                val_losses = []
                for b in range(
                    self.config["train_config"]["num_val_batches_per_epoch"]
                ):
                    loss = self._run_iteration(self.val_gen, False)
                    val_losses.append(loss)
                self.all_val_losses.append(np.mean(val_losses))

                self.info.append(self.all_val_losses[-1])
                print("Validation Loss: %.4f" % self.all_val_losses[-1])

            self.optimizer.param_groups[0]["lr"] = (
                self.initial_lr * (1 - self.epoch / self.max_num_epochs) ** 0.9
            )
            self._finish_online_evaluation()

            epoch_end_time = time()

            if self.epoch % 20 == 0:
                self._save_checkpoint(self, self.all_val_losses[-1], self.epoch)

            self.csv.save(self.info)
            self.info = []

            self._plot_progress()

            self.epoch += 1
            print("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.
        self.csv.end()

    def _save_checkpoint(self, loss, num_epoch):
        if loss < self.best_loss:
            # save checkpoints
            checkpoint = {
                "state_dict": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "loss": loss,
            }
            torch.save(checkpoint, self.output_dir + "/checkpoint.pth.tar")

            self.best_loss = loss

            info_logger.info(
                f"Checkpoint saved on epoch {num_epoch}: loss = {self.best_loss}"
            )
