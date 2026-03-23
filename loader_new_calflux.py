import time as timelib
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import coare35vn
import warnings

from loader_utils_new import *
from data_shapes import *

class WeatherDataset(Dataset):
    """
    Base weather dataset class
    """

    def __init__(
        self,
        device,
        start_date,
        end_date,
        lead_time,
        res=1.5,
        filter_dates=None,
        diff=None,
    ):

        super().__init__()

        # Setup
        self.device = device
        self.data_path = "./mmap_data/"
        self.aux_data_path = "path_to_auxiliary_data/"
        self.start_date = start_date
        self.end_date = end_date
        self.lead_time = lead_time
        self.res = res
        self.filter_dates = filter_dates
        self.diff = diff

        # Date indexing
        self.dates = pd.date_range(start_date, end_date, freq="1d")
        if self.filter_dates == "start":
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month < 7])
        elif self.filter_dates == "end":
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month >= 7])
        else:
            self.index = np.array(range(len(self.dates)))

        # Load the input modalities
        print("Loading ICOADS")
        self.load_icoads()

        print("Loading Drifter")
        self.load_drifter()

        print("Loading Satellite SST L3")
        self.load_satelsst_L3()

        print("Loading Satellite SST L4")
        self.load_satelsst_L4()

        print("Loading GLORYS")
        self.glorys_sfc = [
            self.load_glorys(year)
            for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
        ]
        self.glorys_x = [
            self.to_tensor(np.squeeze(
                np.load(self.data_path + "glorys/glorys_x_{}.npy".format(self.res))
            ))
            / LATLON_SCALE_FACTOR,
            self.to_tensor(np.squeeze(
                np.load(self.data_path + "glorys/glorys_y_{}.npy".format(self.res))
            ))
            / LATLON_SCALE_FACTOR,
        ]

        # Orography
        self.glorys_elev = self.to_tensor(
            np.load(self.data_path + "glorys/glorys_elev.npy")
        )
        self.glorys_elev = torch.flip(self.glorys_elev.permute(0, 2, 1), [-1])
        elev_mean = self.to_tensor(
            np.load(self.data_path + "glorys/mean_glorys_elev.npy")
        )
        elev_std = self.to_tensor(
            np.load(self.data_path + "glorys/std_glorys_elev.npy")
        )
        elev_std = torch.where(elev_std == 0, torch.ones_like(elev_std), elev_std)

        self.glorys_elev = (self.glorys_elev - elev_mean) / elev_std
        self.glorys_elev = torch.nan_to_num(self.glorys_elev, nan=0.0)

        xx, yy = torch.meshgrid(self.glorys_x[0].squeeze(), self.glorys_x[1].squeeze())
        self.glorys_lonlat = torch.stack([xx, yy])

        # Climatology
        self.climatology = np.memmap(self.data_path + "glorys/glorys_climatology.mmap",
                                     dtype="float32",
                                     mode="r",
                                     shape=CLIMATOLOGY_SHAPE,)
        climatology_mean = np.load(
            self.data_path
            + "glorys/glorys_means.npy")[:, np.newaxis, np.newaxis, ...]

        climatology_std = np.load(
            self.data_path
            + "glorys/glorys_stds.npy")[:, np.newaxis, np.newaxis, ...]

        self.climatology = (self.climatology - climatology_mean) / climatology_std
        self.climatology = np.nan_to_num(self.climatology, nan=0.0)

        # Setup normalisation factors
        self.means = np.load(
            self.data_path
            + "glorys/glorys_means.npy")[:, np.newaxis, np.newaxis, ...]
        self.stds = np.load(
            self.data_path
            + "glorys/glorys_stds.npy")[:, np.newaxis, np.newaxis, ...]

    def load_icoads(self):
        """
        Load the ICOADS data
        """

        self.icoads_y = np.memmap(
            self.data_path + "icoads/icoads_1997_2023.mmap",
            dtype="float32",
            mode="r",
            shape=ICOADS_Y_SHAPE,
        )


        self.icoads_x = (
                np.memmap(
                    self.data_path + "icoads/icoads_x.mmap",
                    dtype="float32",
                    mode="r",
                    shape=ICOADS_X_SHAPE,
                )
                / LATLON_SCALE_FACTOR
        )
        self.icoads_means = self.to_tensor(
            np.load(self.data_path + "icoads/mean_icoads.npy")
        )
        self.icoads_stds = self.to_tensor(
            np.load(self.data_path + "icoads/std_icoads.npy")
        )
        self.icoads_index_offset = ICOADS_OFFSETS[self.start_date]

        return

    def load_drifter(self):
        """
        Load the Drifter data
        """

        self.drifter_y = np.memmap(
            self.data_path + "drifter/drifter_2002_2023.mmap",
            dtype="float32",
            mode="r",
            shape = DRIFTER_Y_SHAPE,
        )


        self.drifter_x = (
                np.memmap(
                    self.data_path + "drifter/drifter_x.mmap",
                    dtype="float32",
                    mode="r",
                    shape=DRIFTER_X_SHAPE,
                )
                / LATLON_SCALE_FACTOR
        )

        self.drifter_means = self.to_tensor(
            np.load(self.data_path + "drifter/mean_drifter.npy")
        )
        self.drifter_stds = self.to_tensor(
            np.load(self.data_path + "drifter/std_drifter.npy")
        )
        self.drifter_index_offset = DRIFTER_OFFSETS[self.start_date]

        return

    def load_satelsst_L3(self):
        """
        Load the SATEL-SST-L3 data
        """

        self.satelsst_y_l3 = np.memmap(
            self.data_path + "satelsst_L3/satelsst_2000_2020.mmap",
            dtype="float32",
            mode="r",
            shape=SATELSST_L3_Y_SHAPE,
        )

        xx_l3 = np.load(self.data_path + "satelsst_L3/satelsst_x_1.npy") / LATLON_SCALE_FACTOR
        xx_l3 = np.squeeze(xx_l3)
        yy_l3 = np.load(self.data_path + "satelsst_L3/satelsst_y_1.npy") / LATLON_SCALE_FACTOR
        yy_l3 = np.squeeze(yy_l3)

        self.satelsst_x_l3 = [xx_l3, yy_l3]

        self.satelsst_means_l3 = self.to_tensor(
            np.load(self.data_path + "satelsst_L3/satelsst_means.npy")
        )[:, np.newaxis, np.newaxis, ...]
        self.satelsst_stds_l3 = self.to_tensor(
            np.load(self.data_path + "satelsst_L3/satelsst_stds.npy")
        )[:, np.newaxis, np.newaxis, ...]
        self.satelsst_index_offset_l3 = SATELSST_OFFSETS[self.start_date]

        return
    
    def load_satelsst_L4(self):
        """
        Load the SATEL-SST-L4 data
        """

        self.satelsst_y_l4 = np.memmap(
            self.data_path + "satelsst_L4/satelsst_2000_2020.mmap",
            dtype="float32",
            mode="r",
            shape=SATELSST_L4_Y_SHAPE,
        )

        xx_l4 = np.load(self.data_path + "satelsst_L4/satelsst_x_1.npy") / LATLON_SCALE_FACTOR
        xx_l4 = np.squeeze(xx_l4)
        yy_l4 = np.load(self.data_path + "satelsst_L4/satelsst_y_1.npy") / LATLON_SCALE_FACTOR
        yy_l4 = np.squeeze(yy_l4)

        self.satelsst_x_l4 = [xx_l4, yy_l4]

        self.satelsst_means_l4 = self.to_tensor(
            np.load(self.data_path + "satelsst_L4/satelsst_means.npy")
        )[:, np.newaxis, np.newaxis, ...]
        self.satelsst_stds_l4 = self.to_tensor(
            np.load(self.data_path + "satelsst_L4/satelsst_stds.npy")
        )[:, np.newaxis, np.newaxis, ...]
        self.satelsst_index_offset_l4 = SATELSST_OFFSETS[self.start_date]

        return
    

    def load_glorys(self, year):
        """
        Load the GLORYS training data
        """

        if year % 4 == 0:
            d = 366
        else:
            d = 365
        levels = 1
        x = 240
        y = 114
        mmap = np.memmap(self.data_path + "/glorys/glorys_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, levels, x, y),
                        )
        #assert npy.shape == (d, levels, x, y), f"形状不匹配！期望 {(d, levels, x, y)}，实际 {npy.shape}"
        return mmap

    def norm_glorys(self, x):

        x = (x - self.means) / self.stds
        return x

    def unnorm_glorys(self, x):

        x = x * self.stds + self.means
        return x

    def norm_data(self, x, means, stds):
        return (x - means) / stds

    def __len__(self):
        return self.index.shape[0] - 1

    def to_tensor(self, arr):
        return torch.from_numpy(arr).float().to(self.device)

    def get_time_aux(self, current_date):
        """
        Return the auxiliary temporal channels given a date
        """

        doy = current_date.dayofyear
        year = (current_date.year - 2000) / 1

        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                year,
            ]
        )



class WeatherDatasetAssimilation(WeatherDataset):
    """
    Encoder training loader
    """

    def __init__(
        self,
        device,
        start_date,
        end_date,
        lead_time,
        res=1.5,
        filter_dates=None,
        var_start=0,
        var_end=5,
        diff=False,
        two_frames=False,
    ):

        super().__init__(
            device,
            start_date,
            end_date,
            lead_time,
            res=res,
            filter_dates=filter_dates,
            diff=diff,
        )

        # Setup

        self.var_start = var_start
        self.var_end = var_end
        self.diff = diff
        self.two_frames = two_frames

    def load_glorys_time(self, index):
            """
            GLORYS ground truth data loading
            """

            date = self.dates[index]
            year = date.year
            doy = date.dayofyear - 1

            glorys = self.glorys_sfc[year - int(self.start_date[:4])][doy, ...]
            glorys = np.copy(glorys)
            glorys = self.norm_glorys(glorys[np.newaxis, ...])[0, ...]#这里会不会导致维度不匹配
            return glorys

    def load_year_end(self, year, doy):
        data_1 = self.glorys_sfc[year - int(self.start_date[:4])][doy: doy + 1, ...]
        missing = self.lead_time - data_1.shape[0] + 1
        data_2 = self.glorys_sfc[year - int(self.start_date[:4]) + 1][:missing, ...]
        data = np.concatenate([data_1, data_2])
        return data


    def load_glorys_slice(self, index):
        """
        GLORYS ground truth data loading
        """

        date = self.dates[index]
        year = date.year
        doy = date.dayofyear - 1

        next_date = self.dates[index + 1]
        next_year = next_date.year

        if next_year != year:
            glorys = self.load_year_end(year, doy)
        else:
            glorys = self.glorys_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]

        glorys = self.norm_glorys(np.copy(glorys))
        return glorys


    def __getitem__(self, index):

        if self.two_frames:
            # Case 1: loading t=0 and t=-1
            index = index + 1
            current = self.get_index(index, "current")
            prev = self.get_index(index - 1, "prev")
            current["y_target"] = current["y_target_current"]

            return {**current, **prev}
        else:
            # Case 2: loading t=0
            current = self.get_index(index, "current")
            current["y_target"] = current["y_target_current"]

            return {**current}


    def unnorm_pred(self, x):
        dev = x.device
        x = x.detach().cpu().numpy()

        x = (
            x
            * self.stds[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
            + self.means[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
        )
        return torch.from_numpy(x).float().to(dev)



    def get_index(self, index, prefix):
        """
        Load data for the relevant index respecting different offsets depending on the modality
        """

        index = self.index[index]
        date = self.dates[index]

        # ICOADS
        icoads_x = self.icoads_x[index + self.icoads_index_offset, ...]
        icoads_y = self.icoads_y[index + self.icoads_index_offset, ...]
        icoads_x = [icoads_x[0, :], icoads_x[1, :]]
        icoads_x = [self.to_tensor(i) for i in icoads_x]
        icoads_y = self.to_tensor(icoads_y)
        icoads_y = self.norm_data(icoads_y, self.icoads_means, self.icoads_stds)

        # DRIFTER
        drifter_x = self.drifter_x[index + self.drifter_index_offset, ...]
        drifter_y = self.drifter_y[index + self.drifter_index_offset, ...]
        drifter_x = [drifter_x[0, :], drifter_x[1, :]]
        drifter_x = [self.to_tensor(i) for i in drifter_x]
        drifter_y = self.to_tensor(drifter_y)
        drifter_y = self.norm_data(drifter_y, self.drifter_means, self.drifter_stds)

        # SATELSST_L3
        satelsst_y_l3 = self.to_tensor(self.satelsst_y_l3[index + self.satelsst_index_offset_l3, ...])
        satelsst_x_l3 = [self.to_tensor(i) for i in self.satelsst_x_l3]
        satelsst_y_l3 = self.norm_data(satelsst_y_l3, self.satelsst_means_l3, self.satelsst_stds_l3)
        satelsst_y_l3 = torch.nan_to_num(satelsst_y_l3, nan=0.0)

        # SATELSST_L4
        satelsst_y_l4 = self.to_tensor(self.satelsst_y_l4[index + self.satelsst_index_offset_l4, ...])
        satelsst_x_l4 = [self.to_tensor(i) for i in self.satelsst_x_l4]
        satelsst_y_l4 = self.norm_data(satelsst_y_l4, self.satelsst_means_l4, self.satelsst_stds_l4)
        satelsst_y_l4 = torch.nan_to_num(satelsst_y_l4, nan=0.0)
        

        # GLORYS
        glorys = self.to_tensor(self.load_glorys_time(index))
        glorys_target = glorys.permute(2, 1, 0)
        glorys_target = torch.nan_to_num(glorys_target, nan=0.0)
        glorys_x = self.glorys_x

        # AUxiliary variables
        aux_time = self.to_tensor(self.get_time_aux(date))
        doy_clima = min(date.dayofyear - 1, 364)
        climatology = self.climatology[doy_clima, ...]#这里要修改

        task = {
            "climatology_{}".format(prefix): self.to_tensor(climatology),
            "icoads_x_{}".format(prefix): icoads_x,
            "icoads_{}".format(prefix): icoads_y,
            "drifter_x_{}".format(prefix): drifter_x,
            "drifter_{}".format(prefix): drifter_y,
            "satelsst_l3_{}".format(prefix): satelsst_y_l3,
            "satelsst_l3_x_{}".format(prefix): satelsst_x_l3,
            "satelsst_l4_{}".format(prefix): satelsst_y_l4,
            "satelsst_l4_x_{}".format(prefix): satelsst_x_l4,
            "y_target_{}".format(prefix): glorys_target[
                ..., self.var_start : self.var_end
            ],
            "glorys_x_{}".format(prefix): glorys_x,
            "glorys_elev_{}".format(prefix): self.glorys_elev,
            "glorys_lonlat_{}".format(prefix): self.glorys_lonlat,
            "aux_time_{}".format(prefix): aux_time,
            "lt": torch.Tensor([self.var_start]),
        }

        return task



class ForecastLoader(Dataset):
    """
    Loader for finetuning the processor module
    """

    def __init__(
        self,
        device,
        mode,
        lead_time,
        res=5,
        norm=True,
        diff=False,
        rollout=False,
        random_lt=False,
        ic_path=None,
        finetune_step=None,
        finetune_eval_every=100,
        eval_steps=False,
    ):

        super().__init__()

        # Setup
        self.device = device
        self.mode = mode
        self.data_path = "./mmap_data/"
        self.lead_time = lead_time
        self.res = res
        self.norm = norm
        self.diff = diff
        self.rollout = rollout
        self.random_lt = random_lt
        self.ic_path = ic_path

        self.finetune_step = finetune_step
        self.finetune_eval_every = finetune_eval_every
        self.eval_steps = eval_steps

        freq = "1D"
        if self.mode == "train":
            self.dates = pd.date_range("2000-01-01", "2018-12-31", freq=freq)
        elif self.mode == "tune":
            self.dates = pd.date_range("2020-01-01", "2020-12-31", freq=freq)
        elif self.mode == "test":
            self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)
        elif self.mode == "val":
            self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)
        # Load the predictions from the previous leadtime to be the new context set
        if self.finetune_step is not None:

            if self.mode == "train":
                self.dates = pd.date_range("2000-01-02", "2019-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1)),
                    114,
                    240,
                    3,
                )
            elif self.mode == "val":
                self.dates = pd.date_range("2020-01-01", "2020-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1)),
                    114,
                    240,
                    3,
                )
            elif self.mode == "test":
                self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1)),
                    114,
                    240,
                    3,
                )

            if self.finetune_step > 1:
                print(ic_shape)
                
                self.ic = np.memmap(
                    self.ic_path
                    + "ic_{}_{}.mmap".format(self.mode, self.finetune_step - 1),
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )
            elif self.ic_path is not None:
                print(ic_shape)
                self.ic = np.memmap(
                    self.ic_path + "ic_{}.mmap".format(self.mode),
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )

        elif self.ic_path is not None:
            if self.mode == "train":
                self.dates = pd.date_range("2007-01-02", "2017-12-31", freq=freq)
            ic_shape = (len(self.dates), 114, 240, 24)

            self.ic = np.memmap(
                self.ic_path + "/ic_{}.mmap".format(self.mode),
                dtype="float32",
                mode="r",
                shape=ic_shape,
            )

        # Orography
        self.glorys_elev = self.to_tensor(
            np.load(self.data_path + "glorys/glorys_elev.npy")
        )
        #self.glorys_elev = torch.flip(self.glorys_elev.permute(0, 2, 1), [-1])
        #self.glorys_elev = torch.flip(self.glorys_elev.permute(0, 2, 1), dims=[1])
        elev_mean = self.to_tensor(
            np.load(self.data_path + "glorys/mean_glorys_elev.npy")
        )
        elev_std = self.to_tensor(
            np.load(self.data_path + "glorys/std_glorys_elev.npy")
        )
        elev_std = torch.where(elev_std == 0, torch.ones_like(elev_std), elev_std)
        
        

        self.glorys_elev = (self.glorys_elev - elev_mean) / elev_std
        #self.glorys_elev = (self.glorys_elev - elev_min) / (elev_max - elev_min)
        self.glorys_elev = torch.nan_to_num(self.glorys_elev, nan=0.0)
        #print(self.glorys_elev.shape)

        # GLORYS Noramalisation factors
        self.means = (self.to_tensor(
                         np.load(self.data_path + "glorys/glorys_means.npy")
                                    ).unsqueeze(1).unsqueeze(1)
                    )
        self.stds = (self.to_tensor(
                         np.load(self.data_path + "glorys/glorys_stds.npy")
                                   ).unsqueeze(1).unsqueeze(1)
                    )

        # ERA5 Noramalisation factors
        self.era5_u10_means = (self.to_tensor(
                         np.load(self.data_path + "era5/u10/era5_u10_means.npy")
                                    ).unsqueeze(1).unsqueeze(1)
                    )
        self.era5_u10_stds = (self.to_tensor(
                         np.load(self.data_path + "era5/u10/era5_u10_stds.npy")
                                   ).unsqueeze(1).unsqueeze(1)
                    )

        self.era5_v10_means = (self.to_tensor(
                         np.load(self.data_path + "era5/v10/era5_v10_means.npy")
                                    ).unsqueeze(1).unsqueeze(1)
                    )
        self.era5_v10_stds = (self.to_tensor(
                         np.load(self.data_path + "era5/v10/era5_v10_stds.npy")
                                   ).unsqueeze(1).unsqueeze(1)
                    )

        self.era5_t2m_means = (self.to_tensor(
                         np.load(self.data_path + "era5/t2m/era5_t2m_means.npy")
                                    ).unsqueeze(1).unsqueeze(1)
                    )
        self.era5_t2m_stds = (self.to_tensor(
                         np.load(self.data_path + "era5/t2m/era5_t2m_stds.npy")
                                   ).unsqueeze(1).unsqueeze(1)
                    )

        self.era5_sh_means = (self.to_tensor(
                         np.load(self.data_path + "era5/sh/era5_sh_means.npy")
                                    ).unsqueeze(1).unsqueeze(1)
                    )
        self.era5_sh_stds = (self.to_tensor(
                         np.load(self.data_path + "era5/sh/era5_sh_stds.npy")
                                   ).unsqueeze(1).unsqueeze(1)
                    )

        self.era5_means = torch.cat([self.era5_u10_means, self.era5_v10_means, self.era5_t2m_means, self.era5_sh_means], dim=0)
        self.era5_stds = torch.cat([self.era5_u10_stds, self.era5_v10_stds, self.era5_t2m_stds, self.era5_sh_stds], dim=0)

        self.flux_means = (self.to_tensor(np.load(self.data_path + "era5/cal_flux_lead0/era5_flux_means.npy")).unsqueeze(1).unsqueeze(1))
        self.flux_stds = (self.to_tensor(np.load(self.data_path + "era5/cal_flux_lead0/era5_flux_stds.npy")).unsqueeze(1).unsqueeze(1))

        self.era5_lat = np.load(self.data_path + "era5/t2m/era5_y_1.5.npy")
        self.era5_lat = self.era5_lat.flatten()
        self.lat_grid = np.tile(self.era5_lat.reshape(1, 114), (240, 1))
        self.lat_flat = self.lat_grid.flatten()

                    
        # Diff means and stds
        self.diff_means = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "glorys/mean_glorys_diff.npy")
                ).unsqueeze(1).unsqueeze(1)
            )

        self.diff_stds = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "glorys/std_glorys_diff.npy")
            ).unsqueeze(1).unsqueeze(1)
        )
        #print(f"均值形状:{self.diff_means.shape}")
        
        self.means_dict = {0: self.diff_means}
        self.stds_dict = {0: self.diff_stds}

        # ERA5 and GLORYS ground truth data for training
        self.glorys_sfc = [
            self.load_glorys(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]

        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]
        self.era5fluxlead0_sfc = [
            self.load_era5flux_lead0(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]
        self.era5fluxlead1_sfc = [
            self.load_era5flux_lead1(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]
        

    def __len__(self):

        return self.dates.shape[0] - 1 - 1

    def to_tensor(self, arr):

        return torch.from_numpy(arr).float().to(self.device)

    def norm_glorys(self, x):
        x = (x - self.means) / self.stds
        #x = (x - self.mins) / (self.maxs - self.mins)
        return x


    def norm_glorys_tendency(self, x, lt_offset):

        x = (x - self.means_dict[lt_offset]) / self.stds_dict[lt_offset]
        #x = (x - self.mins_dict[lt_offset]) / (self.maxs_dict[lt_offset] - self.mins_dict[lt_offset])
        return x

    def norm_era5(self, x):
      x = (x - self.era5_means) / self.era5_stds
      return x

    def norm_flux(self, x):
        """归一化计算出的通量"""
        x = (x - self.flux_means) / self.flux_stds
        return x

    def unnorm_pred(self, x):
        x = x * self.diff_stds.permute(2, 1, 0) + self.diff_means.permute(2, 1, 0)
        #x = x * (self.diff_maxs.unsqueeze(0) - self.diff_mins.unsqueeze(0)) + self.diff_mins.unsqueeze(0)
        return x

    def unnorm_base_context(self, x):
        x = x * self.stds + self.means
        #x = x * (self.maxs.unsqueeze(0) - self.mins.unsqueeze(0)) + self.mins.unsqueeze(0)
        return x

    def load_glorys(self, year):
        """
        Load the GLORYS training data
        """

        if year % 4 == 0:
            d = 366
        else:
            d = 365
        levels = 4
        x = 240
        y = 114
        mmap = np.memmap(self.data_path + "/glorys/glorys_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, levels, x, y),
                        )
        return mmap

    

    def load_era5(self, year):
        """
        Load the ERA5 training data
        """

        if year % 4 == 0:
            d = 366
        else:
            d = 365
        x = 240
        y = 114
        era5_u10 = np.memmap(self.data_path + "/era5/u10/era5_u10_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, 1, x, y),
                      )
        era5_v10 = np.memmap(self.data_path + "/era5/v10/era5_v10_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, 1, x, y),
                      )

        era5_t2m = np.memmap(self.data_path + "/era5/t2m/era5_t2m_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, 1, x, y),
                      )
                      

        era5_sh = np.memmap(self.data_path + "/era5/sh/era5_sh_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, 1, x, y),
                      )
                      

        era5_ssrd = np.memmap(self.data_path + "/era5/ssrd/era5_ssrd_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, 1, x, y),
                      )
        era5_strd = np.memmap(self.data_path + "/era5/strd/era5_strd_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, 1, x, y),
                      )
        era5_tp = np.memmap(self.data_path + "/era5/tp/era5_tp_{}.mmap".format(year),
                        dtype="float32",
                        mode="r",
                        shape=(d, 1, x, y),
                      )

  
        era5_mmap = np.concatenate([era5_u10, era5_v10, era5_t2m, 
                       era5_sh, era5_ssrd, era5_strd, era5_tp], axis=1)
        
        #era5_mmap = np.concatenate([era5_u10, era5_v10, era5_t2m, era5_sh], axis=1)

        return era5_mmap


    def load_era5flux_lead0(self, year):
        """
        Load the ERA5_FLUX training data
        """

        if year % 4 == 0:
            d = 366
        else:
            d = 365
        x = 240
        y = 114
        era5_flux_lead0 = np.memmap(self.data_path + "/era5/cal_flux_lead0/era5_flux_{}.mmap".format(year),
                              dtype="float32",
                              mode="r",
                              shape=(d, 8, x, y),
                              )

        return era5_flux_lead0

    def load_era5flux_lead1(self, year):
        """
        Load the ERA5_FLUX training data
        """

        if year % 4 == 0:
            d = 366
        else:
            d = 365
        x = 240
        y = 114
        era5_flux_lead1 = np.memmap(self.data_path + "/era5/cal_flux_lead1/era5_flux_{}.mmap".format(year),
                              dtype="float32",
                              mode="r",
                              shape=(d, 8, x, y),
                              )

        return era5_flux_lead1

    def load_glorys_time(self, index):
            """
            GLORYS ground truth data loading
            """

            date = self.dates[index]
            year = date.year
            doy = date.dayofyear - 1

            glorys = self.glorys_sfc[year - int(self.dates[0].year)][doy, ...]
            #glorys = self.norm_glorys(glorys[np.newaxis, ...])[0, ...]#这里会不会导致维度不匹配
            return np.copy(glorys)

    def load_era5_time(self, index):
            """
            ERA5 ground truth data loading
            """

            date = self.dates[index]
            year = date.year
            doy = date.dayofyear - 1

            era5_raw = self.era5_sfc[year - int(self.dates[0].year)][doy, ...]
            return np.copy(era5_raw)

    def load_era5fluxlead0_time(self, index):
            """
            ERA5 ground truth data loading
            """

            date = self.dates[index]
            year = date.year
            doy = date.dayofyear - 1

            era5_flux_lead0 = self.era5fluxlead0_sfc[year - int(self.dates[0].year)][doy, ...]
            return np.copy(era5_flux_lead0)

    def load_era5fluxlead1_time(self, index):
            """
            ERA5 ground truth data loading
            """

            date = self.dates[index]
            year = date.year
            doy = date.dayofyear - 1

            era5_flux_lead1 = self.era5fluxlead1_sfc[year - int(self.dates[0].year)][doy, ...]
            return np.copy(era5_flux_lead1)

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

    @staticmethod
    def rh_calc(T, P, Q):
        """计算相对湿度"""
        # T: Celsius
        # P: hPa
        # Q: kg/kg
        es = 6.1121 * np.exp(17.502 * T / (T + 240.97)) * (1.0007 + 3.46e-6 * P)
        em = Q * P / (0.378 * Q + 0.622)
        rh_val = 100 * em / es
        return np.clip(rh_val, 0, 100)

    def compute_fluxes(self, era5_np, glorys_np):
        """
        根据 ERA5 和 GLORYS 的 numpy 数组计算通量。
        Input Shapes:
            era5_np: (6, X, Y) -> [u_air, v_air, t2m, sh, sw, lw]
            glorys_np: (5, X, Y) -> [u_sea, v_sea, zos, temp, salt]
        Output:
            flux_tensor: (6, X, Y) on Device -> [tau_u, tau_v, hsb, hlb, rsn, rln]
        """
        _, nlon, nlat = era5_np.shape
        
        # 1. 提取变量并转换单位
        # ERA5
        u_air = era5_np[0].flatten()
        v_air = era5_np[1].flatten()
        t2m_C = era5_np[2].flatten() - 273.15 # K -> C
        sh  = era5_np[3].flatten()
        ssrd = era5_np[4].flatten()
        strd = era5_np[5].flatten()
        tp  = era5_np[6].flatten()


        # GLORYS
        u_sea = glorys_np[0].flatten()
        v_sea = glorys_np[1].flatten()
        sst_C = glorys_np[2].flatten() # C
        
        # 2. 预处理输入
        t2m_C[t2m_C < -100] = np.nan
        sst_C[sst_C < -3.0] = np.nan
        
        du = u_air
        dv = v_air
        spd_mag = np.sqrt(du**2 + dv**2)
        
        # RH 计算
        rh = self.rh_calc(t2m_C, 1015, sh)

        # 3. 运行 COARE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # coare35vn CPU
            A = coare35vn.coare35vn(
                u=spd_mag, t=t2m_C, rh=rh, ts=sst_C,
                Rs=ssrd,Rl=strd,rain=tp,P=1015,zu=10,zt=2, zq=2,
                lat=self.lat_flat,zi=600
            )

        # 4. 后处理结果 (N, 5) -> (N, 5)
        # A columns: [tau, hsb, hlb, rsn, rln]
        tau_total = A[:, 0]
        hsb    = A[:, 1]
        hlb    = A[:, 2]
        rsn    = A[:, 3]
        rln    = A[:, 4]
        evap   = A[:, 5]
        rain_rate = A[:, 6]

        # 矢量分解
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_u = tau_total * (du / spd_mag)
            tau_v = tau_total * (dv / spd_mag)
        
        tau_u[~np.isfinite(tau_u)] = 0.0 # 或 np.nan，并在后面处理
        tau_v[~np.isfinite(tau_v)] = 0.0
        
        # 重塑
        flux_res = np.stack([tau_u, tau_v, hsb, hlb, rsn, rln, evap, rain_rate], axis=0) # (6, N)
        flux_res = flux_res.reshape(8, nlon, nlat)
        
        # 转为 Tensor
        return self.to_tensor(flux_res)
    

    def __getitem__(self, index):
        
        idx_prev = index
        idx_curr = index + 1

        # Option to offset to random leadtime
        lt_offset = 0
        

        # Load ERA5_FLUX ground truth data
        era5flux_prev = self.to_tensor(
            self.load_era5fluxlead0_time(idx_prev)
        )
        era5flux_prev = self.norm_flux(era5flux_prev)
        era5flux_prev = torch.nan_to_num(era5flux_prev, nan=0.0)


        era5flux_curr = self.to_tensor(
            self.load_era5fluxlead0_time(idx_curr)
        )
        era5flux_curr = self.norm_flux(era5flux_curr)
        era5flux_curr = torch.nan_to_num(era5flux_curr, nan=0.0)


        era5flux_futu = self.to_tensor(
            self.load_era5fluxlead1_time(idx_curr)
        )
        era5flux_futu = self.norm_flux(era5flux_futu)
        era5flux_futu = torch.nan_to_num(era5flux_futu, nan=0.0)
        
        '''
        # Load era5_raw
        era5_prev = self.to_tensor(self.load_era5_time(idx_prev))
        era5_prev = self.norm_era5(era5_prev)

        era5_curr = self.to_tensor(self.load_era5_time(idx_curr))
        era5_curr = self.norm_era5(era5_curr)

        era5_futu = self.to_tensor(self.load_era5_time(idx_curr + 1))
        era5_futu = self.norm_era5(era5_futu)
        '''


        
        #y_target = self.to_tensor(
        #    self.load_glorys_time(idx_curr + self.lead_time - lt_offset)
        #)
        #根据截断日期更新进行的修改(相应的__len__也进行了修改)
        y_target = self.to_tensor(
            self.load_glorys_time(idx_curr + 1 - lt_offset)
        )
        
        # Load either initial condition or ERA5 depending on task
        if self.ic_path is not None:
            state_curr = self.to_tensor(self.ic[idx_curr].copy().transpose(2, 1, 0))
            #state_prev = self.to_tensor(self.ic[idx_prev].copy().transpose(2, 1, 0))
        else:
            state_curr = self.to_tensor(self.load_glorys_time(idx_curr))
            state_prev = self.to_tensor(self.load_glorys_time(idx_prev))
        # Auxiliary time
        time = self.to_tensor(self.make_time_channels(idx_curr, state_curr.shape[1], state_curr.shape[2]))
        # Normalisation
        if self.diff:
          
            # Target 也就是 T(target) - T(current)
            y_target = (y_target - state_curr[:4, ...]) 
            y_target = self.norm_glorys_tendency(y_target, lt_offset)
            y_target = y_target.permute(2, 1, 0)
            y_target = torch.nan_to_num(y_target, nan=0.0)
            
            # Context 归一化
            state_curr[:4, ...] = self.norm_glorys(state_curr[:4, ...])
            state_prev[:4, ...] = self.norm_glorys(state_prev[:4, ...])
             
            state_curr = torch.nan_to_num(state_curr, nan=0.0)
            state_prev = torch.nan_to_num(state_prev, nan=0.0)

        else:
            if self.norm:
                state_curr[:4, ...] = self.norm_glorys(state_curr[:4, ...])
                state_curr = torch.nan_to_num(state_curr, nan=0.0)
                state_prev[:4, ...] = self.norm_glorys(state_prev[:4, ...])
                state_prev = torch.nan_to_num(state_prev, nan=0.0)

                y_target = self.norm_glorys(y_target)
                y_target = torch.nan_to_num(y_target, nan=0.0)
            y_target = y_target.permute(2, 1, 0)
        
        y_context = torch.cat([era5flux_prev, state_prev, era5flux_curr, state_curr, era5flux_futu, self.glorys_elev, time], dim=0)
        #使用原始的era5数据
        #y_context = torch.cat([era5_prev, state_prev, era5_curr, state_curr, era5_futu, self.glorys_elev, time], dim=0)
        #不使用任何era5数据
        #y_context = torch.cat([state_prev, state_curr, self.glorys_elev, time], dim=0)
        
        return {
            "y_context": y_context,
            "y_target": y_target[..., :],
            "lt": self.to_tensor(np.array([lt_offset])),
            "target_index": self.to_tensor(np.array([index])),
            }



'''
# 测试代码
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置测试参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建数据集实例
dataset = WeatherDatasetAssimilation(
    device=device,
    start_date='2000-01-01',
    end_date='2001-01-03',
    lead_time=1,
    #era5_mode='sfc',
    res=1.5,
    filter_dates=None,
    var_start=0,
    var_end=1,
    diff=False,
    two_frames=True
)
print(f"数据集创建成功，共有 {len(dataset)} 个样本")

# 测试单个样本
#print("\n=== 单个样本尺寸 ===")
#task = dataset[0]
#for key in sorted(task.keys()):
#    value = task[key]
#    if isinstance(value, torch.Tensor):
#        print(f"{key}: {value.shape}")
#    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
#        shapes = [str(v.shape) for v in value]
#        print(f"{key}: 列表长度{len(value)}, 各tensor尺寸: {shapes}")


# 测试DataLoader batch功能
print("\n=== DataLoader批处理后尺寸 (batch_size=2) ===")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
batch = next(iter(dataloader))

for key in sorted(batch.keys()):
    value = batch[key]
    if isinstance(value, torch.Tensor):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}, type={type(value)}")
    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
        print(f"{key}: 列表长度{len(value)}")
        for i, v in enumerate(value):
            print(f"  [{i}]: shape={v.shape}, dtype={v.dtype}, type={type(v)}")
    else:
        # 其他类型变量也可以简单打印类型
        print(f"{key}: type={type(value)}")

#print(batch['glorys_x_current'][0] * 360)
data_current = batch['glorys_elev_current']
#data_current = torch.flip(data_current.permute(0, 1, 3, 2), [2])
channel_idx=3
data_channel = data_current[0, channel_idx, :, :].cpu().numpy()
plt.figure(figsize=(14, 8))
im = plt.imshow(data_channel, cmap='jet', aspect='auto', origin='upper')
plt.colorbar(im, label=f'Channel {channel_idx} Value')
plt.show()
'''





'''
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# ==========================================
# 2. 开始测试代码
# ==========================================

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建数据集实例
# 注意：mode='train' 会尝试读取 train 对应的日期数据
dataset = ForecastLoader(
    device=device,
    mode='train',  # 模式: train, val, test
    lead_time=1,  # 预测步长
    res=1,  # 分辨率
    norm=False,  # 是否归一化
    diff=True,  # 是否预测差值 (ResNet style)
    rollout=False,  # 是否返回连续序列 (设为 True 可测试 sequence output)
    random_lt=False,  # 是否随机 Lead Time
    ic_path=None,  # 初始条件路径 (None 表示使用 ERA5/Glorys 真值)
    finetune_step=None  # 微调步数
)

print(f"数据集创建成功，共有 {len(dataset)} 个样本")

# -------------------------------------------------
# 测试单个样本
# -------------------------------------------------
print("\n=== 单个样本尺寸 (dataset[0]) ===")
task = dataset[0]
for key in sorted(task.keys()):
    value = task[key]
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape} | dtype={value.dtype}")
    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
        shapes = [str(v.shape) for v in value]
        print(f"{key}: 列表长度{len(value)}, 各tensor尺寸: {shapes}")
    else:
        print(f"{key}: {value}")


# -------------------------------------------------
# 测试 DataLoader Batch 功能
# -------------------------------------------------
print("\n=== DataLoader批处理后尺寸 (batch_size=4) ===")
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(dataloader))

for key in sorted(batch.keys()):
    value = batch[key]
    if isinstance(value, torch.Tensor):
        print(f"{key}: shape={value.shape}, device={value.device}")
    elif isinstance(value, list):
        print(f"{key}: list len={len(value)}")

# -------------------------------------------------
# 可视化测试
# -------------------------------------------------
print("\n=== 可视化检查 ===")
# 获取 Context (t=0) 和 Target (t=LeadTime)
# context shape: [Batch, Channels, H, W]
# 这里的 Channels 包含了 变量 + 地形 + 时间编码
y_context = batch['y_context']
y_target = batch['y_target']



# 提取第 0 个样本，第 0 个通道 (通常是温度或第一个变量)
sample_idx = 1
channel_idx = 0

# y_context 包含辅助变量 (地形/时间)，通常前5个是物理变量
# y_target 只包含预测变量
data_ctx = y_context[sample_idx, channel_idx+3, :, :].cpu().detach().numpy()
data_tgt = y_target[sample_idx, :, :, channel_idx].cpu().detach().numpy()
print(np.nanmax(data_ctx), np.nanmin(data_ctx))


# 计算 Difference (可视化变化趋势)


plt.figure(figsize=(18, 6))

# Plot 1: Context (Input)
plt.subplot(1, 2, 1)
im1 = plt.imshow(data_ctx, cmap='jet', aspect='auto', origin='lower')  # lower origin maps to standard map convention often
plt.colorbar(im1, label='Value')
plt.title(f'Input Context (t=0)\nChannel {channel_idx}', fontsize=12)
plt.xlabel('Longitude Index')
plt.ylabel('Latitude Index')

# Plot 2: Target (Ground Truth)
plt.subplot(1, 2, 2)
im2 = plt.imshow(data_tgt, cmap='jet', aspect='auto', origin='lower')
plt.colorbar(im2, label='Value')
plt.title(f'Target (t={dataset.lead_time} days)\nChannel {channel_idx}', fontsize=12)
plt.xlabel('Longitude Index')



plt.tight_layout()
plt.show()
'''





