import sys

import numpy as np
import torch
import torch.nn as nn
import netCDF4 as nc
from vit_new import *

sys.path.append("../")


class ConvCNPSCS(nn.Module):
    """
    ConvCNP class used for the encoder and processor modules
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            int_channels,
            device,
            data_path="./high_resolution_data/",
    ):

        super().__init__()

        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.data_path = data_path

        # Load internal grid longitude-latitude locations
        with nc.Dataset(self.data_path + "glosuf_GLORYS/glorys_deptho_mdt_122E155E_18N38N.nc", "r") as glorys_constant_file:
            self.glorys_x = ((torch.from_numpy(glorys_constant_file.variables["longitude"][:]) / 360).float().cuda())
            self.glorys_y = ((torch.from_numpy(glorys_constant_file.variables["latitude"][:]) / 360).float().cuda())
            self.mask = (torch.from_numpy(glorys_constant_file.variables["ocean_mask"][:]).float().unsqueeze(
                0).unsqueeze(0).cuda())  # (1,1,241,397)
        self.int_grid = [self.glorys_x.unsqueeze(0), self.glorys_y.unsqueeze(0)]

        # Instantiate the decoder. Here decoder refers to decoder in a convCNP (i.e the ViT backbone)
        self.decoder_lr = ViT(
            in_channels=in_channels,
            out_channels=out_channels,
            h_channels=256,
            depth=8,
            patch_size=4,
            img_size=[241, 397],
            # per_var_embedding=False,
            window_size=7,
        )

        self.break_next = False

    def forward(self, task, film_index):
        # Setup input
        x = task["y_context"]
        # Run ViT backbone
        x = self.decoder_lr(x, lead_times=task["lt"])
        #x = x.permute(0, 3, 1, 2)
        #x = x.permute(0, 2, 3, 1)
        x = x.permute(0, 2, 1, 3)
        mask = self.mask.permute(0, 3, 2, 1)
        return x * mask


if __name__ == "__main__":
    from loader import ForecastLoader
    from torch.utils.data import DataLoader

    # 1. 基础配置
    # 注意：确保 data_path 路径正确
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "./high_resolution_data/"
    print(f"Testing on device: {device}")

    total_in_channels = 38
    target_channels = 5

    # 3. 初始化 Dataset 和 Model
    print("Initializing Loader and Model...")
    dataset = ForecastLoader(device=device, mode="val", lead_time=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = ConvCNPSCS(
        in_channels=total_in_channels,
        out_channels=target_channels,
        int_channels=256,
        device=device,
        data_path=data_path
    ).to(device)

    # 4. 运行单步测试
    print("Fetching batch...")
    batch = next(iter(dataloader))

    # 验证输入维度
    print(f"Input y_context shape: {batch['y_context'].shape}")
    print(f"Input y_target shape:  {batch['y_target'].shape}")
    print("Running Model Forward Pass...")
    try:
        # film_index 在你的代码里虽然在 forward 里定义了但在函数体中没被使用，暂传 None
        preds = model(batch, film_index=None)

        print("\n--- Success! ---")
        print(f"Forward output shape: {preds.shape}")

        # 尺寸检查逻辑
        if preds.shape == batch['y_target'].shape:
            print("Size Flow Check: PASSED (Output matches Target)")
        else:
            print(f"Size Flow Check: FAILED. Output {preds.shape} != Target {batch['y_target'].shape}")

    except Exception as e:
        print(f"\n--- Error during Forward Pass ---")
        print(e)
        # 打印详细错误位置
        import traceback

        traceback.print_exc()

    # 5. 显存清理（可选）
    del model, batch, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()