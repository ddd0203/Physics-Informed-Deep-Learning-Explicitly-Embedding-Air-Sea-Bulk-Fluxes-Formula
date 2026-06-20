import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import netCDF4 as nc
import matplotlib.pyplot as plt


class ForecastLoader(Dataset):
    """
    Loader for finetuning the processor module
    """

    def __init__(
            self,
            device,
            mode,
            lead_time=1,
    ):

        super().__init__()

        # Setup
        self.device = device
        self.mode = mode
        self.data_path = "./high_resolution_data/"
        self.lead_time = lead_time

        freq = "1D"
        if self.mode == "train":
            self.dates = pd.date_range("2000-01-01", "2018-12-31", freq=freq)
        elif self.mode == "test":
            self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)
        elif self.mode == "val":
            self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)

        # Constant shape(170,360)(lat, lon)
        with nc.Dataset(self.data_path + "glosuf_GLORYS/glorys_deptho_mdt_122E155E_18N38N.nc",
                        "r") as glorys_constant_file:
            self.glorys_deptho = self.to_tensor(glorys_constant_file.variables["deptho"][:])
            self.glorys_mdt = self.to_tensor(glorys_constant_file.variables["mdt"][:])

        constant_mean = self.to_tensor(np.load(self.data_path + "glosuf_GLORYS/glorys_constant_mean.npy"))  # (2,1)
        constant_std = self.to_tensor(np.load(self.data_path + "glosuf_GLORYS/glorys_constant_std.npy"))
        self.glorys_deptho = (self.glorys_deptho - constant_mean[0, 0]) / constant_std[0, 0]
        self.glorys_mdt = (self.glorys_mdt - constant_mean[1, 0]) / constant_std[1, 0]
        self.glorys_deptho = torch.nan_to_num(self.glorys_deptho, nan=0.0)
        self.glorys_mdt = torch.nan_to_num(self.glorys_mdt, nan=0.0)

        # GLORYS Noramalisation factors(uo, vo, thetao, so, zos)
        self.glorys_means = self.to_tensor(np.load(self.data_path + "glosuf_GLORYS/glorys_ocean_mean.npy"))  # (5,1)
        self.glorys_stds = self.to_tensor(np.load(self.data_path + "glosuf_GLORYS/glorys_ocean_std.npy"))

        # ERA5 Noramalisation factors(tau_u,tau_v,sensible,latent,net_short,net_long,evap,rain)
        self.flux_means = self.to_tensor(np.load(self.data_path + "bulk_flux/bulk_flux_mean.npy"))  # (8,1)
        self.flux_stds = self.to_tensor(np.load(self.data_path + "bulk_flux/bulk_flux_std.npy"))

        # GLORYS Diff Noramalisation factors
        self.glorys_diff_means = self.to_tensor(
            np.load(self.data_path + "glosuf_GLORYS/glorys_ocean_diff_mean.npy"))  # (5,1)
        self.glorys_diff_stds = self.to_tensor(np.load(self.data_path + "glosuf_GLORYS/glorys_ocean_diff_std.npy"))

        # ERA5 and GLORYS ground truth data for training
        self.glorys_sfc = [
            self.load_glorys(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]
        self.era5fluxes_lead0_sfc = [
            self.load_era5fluxes_curr(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]
        self.era5fluxes_lead1_sfc = [
            self.load_era5fluxes_futu(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]

    def __len__(self):

        # return self.dates.shape[0] - self.lead_time - 1
        # 根据截断日期更新进行的修改
        return self.dates.shape[0] - 1 - self.lead_time

    def to_tensor(self, arr):

        return torch.from_numpy(arr).float().to(self.device)

    def norm_glorys(self, x, var_name):
        """
        var_name: 'uo', 'vo', 'thetao', 'so', 'zos'
        """
        if var_name == 'uo':
            mean = self.glorys_means[0, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_stds[0, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'vo':
            mean = self.glorys_means[1, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_stds[1, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'thetao':
            mean = self.glorys_means[2, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_stds[2, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'so':
            mean = self.glorys_means[3, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_stds[3, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'zos':
            mean = self.glorys_means[4, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_stds[4, 0].unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported variable name: {var_name}")
        return (x - mean) / std

    def norm_era5fluxes(self, x):
        """
        var_name: 'tau_u', 'tau_v', 'sensible', 'latent'
                  'net_short', 'net_long', 'evap', 'rain'
        """
        means = self.flux_means.view(8, 1, 1)
        stds = self.flux_stds.view(8, 1, 1)

        return (x - means) / stds

    def norm_glorys_tendency(self, x, var_name):
        """
        var_name: 'uo', 'vo', 'thetao', 'so', 'zos'
        """
        if var_name == 'uo':
            mean = self.glorys_diff_means[0, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_diff_stds[0, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'vo':
            mean = self.glorys_diff_means[1, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_diff_stds[1, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'thetao':
            mean = self.glorys_diff_means[2, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_diff_stds[2, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'so':
            mean = self.glorys_diff_means[3, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_diff_stds[3, 0].unsqueeze(-1).unsqueeze(-1)
        elif var_name == 'zos':
            mean = self.glorys_diff_means[4, 0].unsqueeze(-1).unsqueeze(-1)
            std = self.glorys_diff_stds[4, 0].unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported variable name: {var_name}")
        return (x - mean) / std

    def unnorm_tendency(self, preds):
        """
        preds: [B, H, W, 5] or [B, 5, H, W]
        variable order: uo, vo, thetao, so, zos
        """
        means = self.glorys_diff_means[:, 0]
        stds = self.glorys_diff_stds[:, 0]

        if preds.ndim == 4 and preds.shape[-1] == 5:
            means = means.view(1, 1, 1, 5).to(preds.device)
            stds = stds.view(1, 1, 1, 5).to(preds.device)
        elif preds.ndim == 4 and preds.shape[1] == 5:
            means = means.view(1, 5, 1, 1).to(preds.device)
            stds = stds.view(1, 5, 1, 1).to(preds.device)
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

        return preds * stds + means

    '''
    def unnorm_base_context(self, x):
        x = x * self.stds + self.means
        # x = x * (self.maxs.unsqueeze(0) - self.mins.unsqueeze(0)) + self.mins.unsqueeze(0)
        return x
    '''

    def load_glorys(self, year):
        """
        Load the GLORYS training data (time, latitude, longitude)
        """
        if year % 4 == 0:
            d = 366
        else:
            d = 365
        x = 241
        y = 397
        with nc.Dataset(f"{self.data_path}glosuf_GLORYS/glorys_{year}_daily_122E155E_18N38N.nc",
                        "r") as glorys_surf_file:
            glorys_uo = glorys_surf_file.variables["uo"][:, 0, :, :]  # (d, x, y)
            glorys_vo = glorys_surf_file.variables["vo"][:, 0, :, :]  # (d, levels, x, y)
            glorys_thetao = glorys_surf_file.variables["thetao"][:, 0, :, :]  # (d, x, y)
            glorys_so = glorys_surf_file.variables["so"][:, 0, :, :]  # (d, x, y)
            glorys_zos = glorys_surf_file.variables["zos"][:]  # (d, x, y)

            assert glorys_uo.shape == (d, x, y), f"uo shape err: {glorys_uo.shape}"
            assert glorys_vo.shape == (d, x, y), f"vo shape err: {glorys_vo.shape}"
            assert glorys_thetao.shape == (d, x, y), f"thetao shape err: {glorys_thetao.shape}"
            assert glorys_so.shape == (d, x, y), f"so shape err: {glorys_so.shape}"
            assert glorys_zos.shape == (d, x, y), f"zos shape err: {glorys_zos.shape}"

        return glorys_uo, glorys_vo, glorys_thetao, glorys_so, glorys_zos

    def load_era5fluxes_curr(self, year):
        """
        Load the ERA5 Fluxes curr training data
        """
        if year % 4 == 0:
            d = 366
        else:
            d = 365
        x = 241
        y = 397

        with nc.Dataset(f"{self.data_path}bulk_flux/Calculated_Flux_{year}_highres.nc", "r") as era5_flux_file:
            flux_tauu = era5_flux_file.variables["tau_u"][:]  # (d, x, y)
            flux_tauv = era5_flux_file.variables["tau_v"][:]  # (d, x, y)
            flux_sens = era5_flux_file.variables["sensible"][:]  # (d, x, y)
            flux_late = era5_flux_file.variables["latent"][:]  # (d, x, y)
            flux_shor = era5_flux_file.variables["net_short"][:]  # (d, x, y)
            flux_long = era5_flux_file.variables["net_long"][:]  # (d, x, y)
            flux_evap = era5_flux_file.variables["evap"][:]  # (d, x, y)
            flux_rain = era5_flux_file.variables["rain"][:]  # (d, x, y)

            stacked_fluxes = np.stack([
                flux_tauu, flux_tauv, flux_sens, flux_late,
                flux_shor, flux_long, flux_evap, flux_rain], axis=1)  # (d, 8, x, y)

        return stacked_fluxes


    def load_era5fluxes_futu(self, year):
        """
        Load the ERA5 Fluxes futu training data
        """
        if year % 4 == 0:
            d = 366
        else:
            d = 365
        x = 241
        y = 397

        with nc.Dataset(f"{self.data_path}bulk_flux_lead1/Calculated_Flux_{year}_Lead1_highres.nc", "r") as era5_flux_file:
            flux_tauu = era5_flux_file.variables["tau_u"][:]  # (d, x, y)
            flux_tauv = era5_flux_file.variables["tau_v"][:]  # (d, x, y)
            flux_sens = era5_flux_file.variables["sensible"][:]  # (d, x, y)
            flux_late = era5_flux_file.variables["latent"][:]  # (d, x, y)
            flux_shor = era5_flux_file.variables["net_short"][:]  # (d, x, y)
            flux_long = era5_flux_file.variables["net_long"][:]  # (d, x, y)
            flux_evap = era5_flux_file.variables["evap"][:]  # (d, x, y)
            flux_rain = era5_flux_file.variables["rain"][:]  # (d, x, y)

            stacked_fluxes = np.stack([
                flux_tauu, flux_tauv, flux_sens, flux_late,
                flux_shor, flux_long, flux_evap, flux_rain], axis=1) # (d, 8, x, y)

        return stacked_fluxes


    def load_glorys_time(self, index):
        """
        GLORYS ground truth data loading
        """
        date = self.dates[index]
        year = date.year
        doy = date.dayofyear - 1
        glorys_uo = self.glorys_sfc[year - int(self.dates[0].year)][0][doy, ...]
        glorys_vo = self.glorys_sfc[year - int(self.dates[0].year)][1][doy, ...]
        glorys_thetao = self.glorys_sfc[year - int(self.dates[0].year)][2][doy, ...]
        glorys_so = self.glorys_sfc[year - int(self.dates[0].year)][3][doy, ...]
        glorys_zos = self.glorys_sfc[year - int(self.dates[0].year)][4][doy, ...]

        return np.copy(glorys_uo), np.copy(glorys_vo), np.copy(glorys_thetao), np.copy(glorys_so), np.copy(glorys_zos)

    def load_era5fluxes_leadtime0(self, index):
        """
        ERA5 ground truth data loading
        """
        date = self.dates[index]
        year = date.year
        doy = date.dayofyear - 1
        day_fluxes = self.era5fluxes_lead0_sfc[year - int(self.dates[0].year)][doy, ...]  # (8, x, y)

        return np.copy(day_fluxes)


    def load_era5fluxes_leadtime1(self, index):
        """
        ERA5 ground truth data loading
        """
        date = self.dates[index]
        year = date.year
        doy = date.dayofyear - 1
        day_fluxes = self.era5fluxes_lead1_sfc[year - int(self.dates[0].year)][doy, ...]#(8, x, y)

        return np.copy(day_fluxes)


    def make_time_channels(self, index, x, y):
        """
        Make auxiliary time channels
        """
        date = self.dates[index]
        doy = date.dayofyear - 1
        if date.year % 4 == 0:
            n_days = 366
        else:
            n_days = 365
        doy_sin = np.sin(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))
        doy_cos = np.cos(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))

        return np.concatenate([doy_sin, doy_cos])

    def __getitem__(self, index):
        idx_prev = index
        idx_curr = index + 1
        idx_target = idx_curr + self.lead_time

        # Option to offset to random leadtime
        lt_offset = self.lead_time
        LATLON_SCALE_FACTOR = 360

        # Load era5_fluxes
        era5_fluxes_prev = self.to_tensor(self.load_era5fluxes_leadtime0(idx_prev))
        era5_fluxes_prev = self.norm_era5fluxes(era5_fluxes_prev)
        era5_fluxes_prev = torch.nan_to_num(era5_fluxes_prev)
        
        era5_fluxes_curr = self.to_tensor(self.load_era5fluxes_leadtime0(idx_curr))
        era5_fluxes_curr = self.norm_era5fluxes(era5_fluxes_curr)
        era5_fluxes_curr = torch.nan_to_num(era5_fluxes_curr)
        
        era5_fluxes_futu = self.to_tensor(self.load_era5fluxes_leadtime1(idx_curr))
        era5_fluxes_futu = self.norm_era5fluxes(era5_fluxes_futu)
        era5_fluxes_futu = torch.nan_to_num(era5_fluxes_futu)

        # Load either initial condition or ERA5 depending on task
        glorys_uo_prev, glorys_vo_prev, glorys_thetao_prev, glorys_so_prev, glorys_zos_prev = [
            self.to_tensor(var) for var in self.load_glorys_time(idx_prev)]

        glorys_uo_curr, glorys_vo_curr, glorys_thetao_curr, glorys_so_curr, glorys_zos_curr = [
            self.to_tensor(var) for var in self.load_glorys_time(idx_curr)]

        y_target_uo, y_target_vo, y_target_thetao, y_target_so, y_target_zos = [
            self.to_tensor(var) for var in self.load_glorys_time(idx_target)]

        y_target_uo_diff = y_target_uo - glorys_uo_curr
        y_target_vo_diff = y_target_vo - glorys_vo_curr
        y_target_thetao_diff = y_target_thetao - glorys_thetao_curr
        y_target_so_diff = y_target_so - glorys_so_curr
        y_target_zos_diff = y_target_zos - glorys_zos_curr

        # Normalisation
        glorys_uo_prev = self.norm_glorys(glorys_uo_prev, 'uo')
        glorys_vo_prev = self.norm_glorys(glorys_vo_prev, 'vo')
        glorys_thetao_prev = self.norm_glorys(glorys_thetao_prev, 'thetao')
        glorys_so_prev = self.norm_glorys(glorys_so_prev, 'so')
        glorys_zos_prev = self.norm_glorys(glorys_zos_prev, 'zos')
        glorys_surf_prev = torch.stack([glorys_uo_prev, glorys_vo_prev,
                                        glorys_thetao_prev, glorys_so_prev, glorys_zos_prev], dim=0)
        glorys_surf_prev = torch.nan_to_num(glorys_surf_prev)

        glorys_uo_curr = self.norm_glorys(glorys_uo_curr, 'uo')
        glorys_vo_curr = self.norm_glorys(glorys_vo_curr, 'vo')
        glorys_thetao_curr = self.norm_glorys(glorys_thetao_curr, 'thetao')
        glorys_so_curr = self.norm_glorys(glorys_so_curr, 'so')
        glorys_zos_curr = self.norm_glorys(glorys_zos_curr, 'zos')
        glorys_surf_curr = torch.stack([glorys_uo_curr, glorys_vo_curr,
                                        glorys_thetao_curr, glorys_so_curr, glorys_zos_curr], dim=0)
        glorys_surf_curr = torch.nan_to_num(glorys_surf_curr)

        y_target_uo_diff = self.norm_glorys_tendency(y_target_uo_diff, 'uo')
        y_target_vo_diff = self.norm_glorys_tendency(y_target_vo_diff, 'vo')
        y_target_thetao_diff = self.norm_glorys_tendency(y_target_thetao_diff, 'thetao')
        y_target_so_diff = self.norm_glorys_tendency(y_target_so_diff, 'so')
        y_target_zos_diff = self.norm_glorys_tendency(y_target_zos_diff, 'zos')
        y_target = torch.stack([y_target_uo_diff, y_target_vo_diff,
                                y_target_thetao_diff, y_target_so_diff, y_target_zos_diff], dim=0)
        y_target = y_target.permute(2, 1, 0)
        y_target = torch.nan_to_num(y_target)

        # Auxiliary time
        time = self.to_tensor(
            self.make_time_channels(idx_target, glorys_thetao_curr.shape[0], glorys_thetao_curr.shape[1]))

        y_context = torch.cat([era5_fluxes_prev, glorys_surf_prev, era5_fluxes_curr, glorys_surf_curr, era5_fluxes_futu,
                               self.glorys_deptho.unsqueeze(0), self.glorys_mdt.unsqueeze(0), time], dim=0)

        return {
            "y_context": y_context,
            "y_target": y_target[..., :],
            "lt": self.to_tensor(np.array([lt_offset])),
            "target_index": self.to_tensor(np.array([idx_target])),
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # 1. 基础设置
    # 为了测试方便，建议在有数据的地方运行。请确保 E:/scs_3d_forecast_data/ 路径下数据完整
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 2. 实例化 Dataset 和 DataLoader
        print("Initializing ForecastLoader (val mode)...")
        dataset = ForecastLoader(device=device, mode="val", lead_time=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # 3. 获取一个 Batch 的数据
        print("Fetching one batch of data...")
        batch = next(iter(dataloader))

        y_context = batch["y_context"]  # Shape: [B, C, X, Y]

        print("y_context shape:", batch["y_context"].shape)
        print("y_target shape:", batch["y_target"].shape)

        # ====== 4. 提取用于可视化的数据 ======
        # 我们挑选 batch 中第一个样本 (index 0)

        # 挑选 y_context 的第 2 个通道 -> 根据拼接顺序，0-1是ERA5风场，2应该是GLORYS前一时刻的SLA
        plot_channel = 8
        # 将 Tensor 转回 numpy，由于可能有 nan，使用 nan_to_num 保证画图不出错
        context_img = y_context[0, plot_channel, :, :].cpu().numpy()
        # context_img = np.nan_to_num(context_img)

        # ====== 5. 绘图 ======
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))

        # 子图 1: y_context 物理场图
        # 因为我们没有对应网格的精确经纬度，这里使用像素坐标系 imshow
        im = axes.imshow(context_img, cmap='viridis', origin='lower')
        axes.set_title(f"y_context (Channel {plot_channel}: SLA prev)\nPixel Grid")
        axes.set_xlabel("X grid")
        axes.set_ylabel("Y grid")
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error occurred during testing: {e}")