import os
import sys
import pickle
import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


from trainer import Trainer
from loss_functions import WeightedRmseLoss, PressureWeightedRmseLoss, RmseLoss
from loader import *
from model import *



torch.set_float32_matmul_precision("medium")



def main(output_dir, args):
    """
    Primary training script for the encoder, processor and decoder modules.
    """

    lead_time = args.lead_time

    # Instantiate loss function

    lf = WeightedRmseLoss(
            start_ind=args.start_ind,
            end_ind=args.end_ind,
            weight_per_variable=bool(args.weight_per_variable)
        )


    # Setup datasets training processor

    train_dataset = ForecastLoader(
                device="cuda",
                mode="train",
                lead_time=lead_time,
            )
    val_dataset = ForecastLoader(
                device="cuda",
                mode="val",
                lead_time=lead_time,
            )


    try:
        os.mkdir(f"{output_dir}")
    except FileExistsError:
        pass

    output_dir = f"{output_dir}/"


    model = ConvCNPSCS(
            in_channels=args.in_channels,
            out_channels=args.end_ind - args.start_ind,
            int_channels=args.int_channels,
            device="cuda",
        )

    # Instantiate loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Instantiate trainer

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        lf,
        output_dir,
        args.lr,
    )

    # Train model

    trainer.train(n_epochs=args.epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./model_test/")#trainer.py中DDPTrainer的参数,save_path,模型输出结果的路径
    parser.add_argument("--in_channels", type=int, default=30)#model_new的参数,vit模块的输入通道数量
    parser.add_argument("--out_channels", type=int, default=5)#model_new的参数,vit模块的输出通道数量
    parser.add_argument("--int_channels", type=int, default=256)#model_new的参数,vit模块的中间隐藏通道数量
    parser.add_argument("--loss", default="lw_rmse")#选取loss函数的类型
    parser.add_argument("--weight_per_variable", type=int, default=0)#loss函数的参数,是否引入变量权重
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lead_time", type=int, default=1)#向前预报时间
    parser.add_argument("--weight_decay", type=float, default=1e-6)#trainer中的参数,貌似未使用
    parser.add_argument("--start_ind", type=int, default=0)#开始预报变量的索引
    parser.add_argument("--end_ind", type=int, default=5)#结束预报变量的索引
    args = parser.parse_args()

    torch.device("cuda")
    assert args.lead_time >= 1, "lead_time must be >= 1 for tendency forecasting."
    # Create results directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save config
    with open(output_dir + "/config.pkl", "wb") as f:
        pickle.dump(vars(args), f)
    main(output_dir, args)