import sys

import numpy as np
import torch
import torch.nn as nn

from architectures import MLP
from set_convs import convDeepSet
from unet_wrap_padding import *
from vit import *

sys.path.append("../")


class ConvCNPWeather(nn.Module):
    """
    ConvCNP class used for the encoder and processor modules
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        int_channels,
        device,
        res,
        data_path="./mmap_data/",
        gnp=False,
        mode="assimilation",
        decoder="vit_assimilation",
        film=False,
        two_frames=False,
    ):

        super().__init__()

        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.decoder = decoder
        self.int_x = 256
        self.int_y = 128
        self.data_path = data_path
        self.mode = mode
        self.film = film
        self.two_frames = two_frames


        N_ICOADS_VARS = 1
        N_DRIFTER_VARS = 1
        N_ARGO_VARS = 1

        self.mask = torch.from_numpy(
            np.load(data_path + "/loss_weights/glorys_mask.npy")[
                np.newaxis, ..., np.newaxis
            ]
        ).float().cuda()

        # Load internal grid longitude-latitude locations
        self.glorys_x = ((
            torch.from_numpy(
                np.load(self.data_path + "glorys/glorys_x_{}.npy".format(res))
            ) / 360).float().cuda()
        )
        self.glorys_y = ((
            torch.from_numpy(
                np.load(self.data_path + "glorys/glorys_y_{}.npy".format(res))
            ) / 360).float().cuda()
        )

        self.int_grid = [self.glorys_x, self.glorys_y]
        #self.int_grid = [self.int_grid[0].unsqueeze(0), self.int_grid[1].unsqueeze(0)]

        # Create input setconvs for each data modality
        self.satelsst_setconvs_l3 = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(1)
        ]

        self.satelsst_setconvs_l4 = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(1)
        ]
      


        self.icoads_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_ICOADS_VARS)
        ]
        self.drifter_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_DRIFTER_VARS)
        ]
        

        self.sc_out = convDeepSet(
            0.001, "OnToOff", density_channel=False, device=self.device
        )

        # Instantiate the decoder. Here decoder refers to decoder in a convCNP (i.e the ViT backbone)
        if self.decoder == "vit":
            self.decoder_lr = ViT(
                in_channels=in_channels,
                out_channels=out_channels,
                h_channels=128,
                depth=4,
                patch_size=3,
                per_var_embedding=False,
                img_size=[240, 114],
            )

        elif self.decoder == "vit_assimilation":
            self.decoder_lr = ViT(
                in_channels=24,
                out_channels=out_channels,
                h_channels=128,
                depth=4,
                patch_size=3,
                per_var_embedding=False,
                img_size=[256, 128],
            )

        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            h_channels=128,
            h_layers=4,
        )
        self.break_next = False


    def encoder_icoads(self, task, prefix):
        """
        Data preprocessing for ICOADS
        """

        encodings = []
        for channel in range(1):
            encodings.append(
                self.icoads_setconvs[channel](
                    x_in=task["icoads_x_{}".format(prefix)],
                    wt=task["icoads_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
            #print(task["icoads_{}".format(prefix)][:, channel, :].unsqueeze(1))
        encodings = torch.cat(encodings, dim=1)

        return encodings

    def encoder_drifter(self, task, prefix):
        """
        Data preprocessing for DRIFTER
        """

        encodings = []
        for channel in range(1):
            encodings.append(
                self.drifter_setconvs[channel](
                    x_in=task["drifter_x_{}".format(prefix)],
                    wt=task["drifter_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)


        return encodings


    def encoder_satelsst_l3(self, task, prefix):
        """
        Data preprocessing for SATELSST-L3
        """

        encodings = []
        #task["amsua_{}".format(prefix)][..., -1] = np.nan
        #task["amsua_{}".format(prefix)][task["amsua_{}".format(prefix)] == 0] = np.nan
        for i in range(1):
            encodings.append(
                self.satelsst_setconvs_l3[i](
                    x_in=task["satelsst_l3_x_{}".format(prefix)],
                    wt=task["satelsst_l3_{}".format(prefix)][:, i : i + 1, ...],
                    x_out=self.int_grid,
                )
            )
            #print(task["satelsst_{}".format(prefix)][:, i : i + 1, ...])

        encodings = torch.cat(encodings, dim=1)

        return encodings

    
    def encoder_satelsst_l4(self, task, prefix):
        """
        Data preprocessing for SATELSST-L4
        """

        encodings = []
        #task["amsua_{}".format(prefix)][..., -1] = np.nan
        #task["amsua_{}".format(prefix)][task["amsua_{}".format(prefix)] == 0] = np.nan
        for i in range(1):
            encodings.append(
                self.satelsst_setconvs_l4[i](
                    x_in=task["satelsst_l4_x_{}".format(prefix)],
                    wt=task["satelsst_l4_{}".format(prefix)][:, i : i + 1, ...],
                    x_out=self.int_grid,
                )
            )
            #print(task["satelsst_{}".format(prefix)][:, i : i + 1, ...])

        encodings = torch.cat(encodings, dim=1)

        return encodings
    




    def forward(self, task, film_index):

        # Setup input
        if self.mode == "assimilation":

            elev = torch.flip(task["glorys_elev_current"].permute(0, 1, 3, 2), dims=[2])


            if not self.two_frames:
                encodings = [
                    self.encoder_icoads(task, "current"),
                    self.encoder_drifter(task, "current"),
                    self.encoder_satelsst_l3(task, "current"),
                    self.encoder_satelsst_l4(task, "current"),
                    elev,
                    task["climatology_current"],
                    torch.ones_like(elev[:, :3, ...])
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
            else:
                # Option to pass two timesteps (t=-1 and t=0) as input
                encodings = [
                    self.encoder_icoads(task, "current"),
                    self.encoder_drifter(task, "current"),
                    self.encoder_satelsst_l3(task, "current"),
                    self.encoder_satelsst_l4(task, "current"),
                    self.encoder_icoads(task, "prev"),
                    self.encoder_drifter(task, "prev"),
                    self.encoder_satelsst_l3(task, "prev"),
                    self.encoder_satelsst_l4(task, "prev"),
                    elev,
                    task["climatology_current"],
                    torch.ones_like(elev[:, :3, ...])
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
            x = torch.cat(encodings, dim=1)

        else:
            x = task["y_context"]

        if x.shape[-1] > x.shape[-2]:
            x = x.permute(0, 1, 3, 2)



        # Run ViT backbone
        if self.decoder == "vit":
            x = self.decoder_lr(x, lead_times=task["lt"])
            x = x.permute(0, 3, 1, 2)
        else:
            x = nn.functional.interpolate(x, size=(256, 128))
            x = self.decoder_lr(x, film_index=(task["lt"] * 0) + 1)

        # Process outputs

        if np.logical_and(
            self.mode == "assimilation", self.decoder == "vit_assimilation"
        ):
            x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(240, 114))
            return x.permute(0, 3, 2, 1) * self.mask

        elif self.mode == "forecast":
            x = nn.functional.interpolate(x, size=(240, 114)).permute(0, 2, 3, 1)
            return x.permute(0, 2, 1, 3) * self.mask

        return x

'''
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from loader_new import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据集实例（确保你已经正确导入WeatherDatasetAssimilation）
    
   
    dataset2 = ForecastLoader(
        device=device,
        mode="train",
        lead_time=1,
        res=1.5,
        norm=True,
        diff=True,
        rollout=False,
        random_lt=False,
        ic_path=None,
        finetune_step=None,
        finetune_eval_every=100,
        eval_steps=False,
    )
    print(f"数据集创建成功，共有 {len(dataset)} 个样本")

    # 构造DataLoader，batch大小设为1方便调试
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False)

    # 实例化模型，参数根据你的定义填入
    model = ConvCNPWeather(
        in_channels=20,              # 注意这里和你模型中decoder对应
        out_channels=1,             # 根据你的预测变量数量调整
        int_channels=512,
        device=device,
        res=1.5,
        decoder="vit_assimilation",  # 选择对应decoder
        mode="assimilation",
        two_frames=True,
    )
    model.to(device)
    model.eval()

    model2 = ConvCNPWeather(
        in_channels=7,
        out_channels=1,
        int_channels=512,
        device=device,
        res=1.5,
        decoder="vit",
        mode="forecast",
    )
    model2.to(device)
    model2.eval()

    # 取一个batch输入并推理
    batch = next(iter(dataloader2))

    # 确保batch中的所有tensor都迁移到设备
    batch_on_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_on_device[k] = v.to(device)
        else:
            batch_on_device[k] = v

    with torch.no_grad():
        output = model2(batch_on_device, film_index=None)

    # 输出Tensor形状
    print("模型输出shape:", output.shape)

    # 简单可视化第一个样本的第一个通道的预测（根据具体通道调整）
    img = output[0].squeeze().cpu().numpy()
    #print(img)
    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    #plt.title("模型预测示例(第1通道)")
    plt.show()
  '''


