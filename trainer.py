import sys
import subprocess

import numpy as np
import torch
from tqdm import tqdm
from model import *
from loss_functions import *

sys.path.append("..")


class Trainer:
    """
    Main class for training models using DDP
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            loss_function,
            save_path,
            learning_rate,
            device='cuda',
            test_loader=None,
    ):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.loss_function = loss_function
        self.best_loss = 1000
        self.test_loader = test_loader

        self.model = self.model.to(device)

        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, 433 * 140)

        self.losses = []
        self.train_losses = []
        self.maes = []

    def _unravel_to_numpy(self, x):
        return x.view(-1).detach().cpu().numpy()

    def eval_epoch(self, expand):

        self.model.eval()
        lf = []
        with torch.no_grad():
            for count, task in enumerate(self.val_loader):

                out = self.model(task, film_index=1.0)

                prev_step = None
                l = (
                    self.loss_function(
                        task["y_target"], out, expand=expand
                    )
                    .detach()
                    .cpu().numpy()
                )
                lf.append(l)

        per_var_loss = np.mean(np.array(lf), axis=0)
        np.save(self.save_path + f"per_variable_loss_epoch_{self.epoch}.npy",
                per_var_loss)
        log_loss = np.nanmean(per_var_loss)

        if log_loss < self.best_loss:
            # print(f"unnorm_pred形状: {out.shape}")
            np.save(
                self.save_path + "unnorm_preds.npy",
                self.train_loader.dataset.unnorm_tendency(out).detach().cpu().numpy(),
            )
            np.save(
                self.save_path + "unnorm_targets.npy",
                self.train_loader.dataset.unnorm_tendency(task["y_target"])
                .detach()
                .cpu()
                .numpy(),
            )
        if self.epoch % 5 == 0:
            np.save(self.save_path + "preds_eval.npy", out.cpu().numpy())
            np.save(
                self.save_path + "y_target_eval.npy", task["y_target"].cpu().numpy()
            )

        return log_loss

    def train(self, n_epochs=100):

        expand = False
        prev_step = None
        self.epoch = 0

        for epoch in range(n_epochs):
            self.epoch = epoch
            self.model.train()
            train_loss = []
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for count, task in enumerate(tepoch):

                    out = self.model(task, film_index=1.0)

                    loss = self.loss_function(
                        task["y_target"], out, expand=expand
                    )

                    loss.backward()
                    tepoch.set_postfix(loss=loss.item())
                    prev_step = out

                    self.opt.step()
                    self.opt.zero_grad()
                    train_loss.append(loss.item())
                    if epoch > 10:
                        self.scheduler.step()

            epoch_loss = self.eval_epoch(expand=True)  # epoch_loss:loss函数计算的误差;log_loss_unnorm:反归一化之后计算的loss误差
            train_loss = np.mean(train_loss)  # 训练误差

            self.losses.append(epoch_loss)
            self.train_losses.append(train_loss)
            np.save(
                self.save_path + "losses.npy",
                np.array(self.losses),  # 验证集loss误差
            )
            np.save(
                self.save_path + "train_losses.npy",
                np.array(self.train_losses),  # 训练集loss误差
            )
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.opt.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "loss": epoch_loss,
                    },
                    self.save_path + "epoch_{}".format(epoch),
                )

                try:
                    np.save(
                        self.save_path + "preds_train.npy",
                        out.detach().cpu().numpy(),
                    )
                    np.save(
                        self.save_path + "y_target_train.npy",
                        task["y_target"].detach().cpu().numpy(),
                    )
                except:
                    pass