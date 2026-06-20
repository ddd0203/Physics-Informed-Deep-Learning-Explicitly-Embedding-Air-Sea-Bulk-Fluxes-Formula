"""
Real application forecast from one GLO12v4_ERA5_YYYYMMDD_12days_on_target_grid.nc file.

Input file example:
    F:/GLO12v4预报数据/GLO12v4_ERA5_20260412_12days_on_target_grid.nc

Input file contains:
    ocean state variables:
        uo, vo, thetao, so, zos
    ERA5 atmospheric variables:
        u10, v10, t2m, sh, ssrd, strd, tp

Forecast setting:
    Use first two ocean states:
        day 0 = O(t-1)
        day 1 = O(t)
    Forecast next 10 days:
        day 2 ... day 11

Output:
    NetCDF file with predicted:
        uo, vo, thetao, so, zos
    for 10 forecast days.

Important:
    This script does NOT compute RMSE.
    This script does NOT read validation truth from ForecastLoader.
    This script does NOT read precomputed Calculated_Flux_*.nc.
    It computes bulk flux online from predicted ocean state and ERA5 forcing.
"""

import os
import re
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import torch
import netCDF4 as nc

import coare35vn
from model import ConvCNPSCS


# ============================================================
# Variable definitions
# ============================================================

VAR_NAMES = ["uo", "vo", "thetao", "so", "zos"]

ATM_NAMES = ["u10", "v10", "t2m", "sh", "ssrd", "strd", "tp"]

FLUX_NAMES = [
    "tau_u", "tau_v", "sensible", "latent",
    "net_short", "net_long", "evap", "rain",
]


# ============================================================
# Basic helpers
# ============================================================

def to_torch(arr, device):
    """Convert numpy / masked array to torch tensor, preserving NaN."""
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr, dtype=np.float32)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def as_np_float32(arr):
    """Convert masked array to np.float32 with NaN."""
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=np.float32)


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def parse_start_date_from_filename(path):
    """
    Parse YYYYMMDD from:
        GLO12v4_ERA5_20260412_12days_on_target_grid.nc
    """
    base = os.path.basename(path)
    m = re.search(r"GLO12v4_ERA5_(\d{8})_\d+days_on_target_grid\.nc$", base)
    if m is None:
        return None
    return pd.Timestamp(m.group(1), tz=None)


def nc_time_to_pandas_dates(ds, fallback_start_date=None):
    """
    Decode NetCDF time variable to pandas Timestamp list.
    If decoding fails, use filename-derived fallback_start_date.
    """
    if "time" not in ds.variables:
        if fallback_start_date is None:
            raise KeyError("No time variable and no fallback start date.")
        return [fallback_start_date + pd.Timedelta(days=i) for i in range(12)]

    time_var = ds.variables["time"]
    raw = time_var[:]
    ntime = len(raw)

    try:
        units = getattr(time_var, "units")
        calendar = getattr(time_var, "calendar", "standard")
        decoded = nc.num2date(
            raw,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
        )

        dates = []
        for d in decoded:
            dates.append(
                pd.Timestamp(
                    year=d.year,
                    month=d.month,
                    day=d.day,
                    hour=getattr(d, "hour", 0),
                    minute=getattr(d, "minute", 0),
                    second=int(getattr(d, "second", 0)),
                )
            )
        return dates

    except Exception:
        if fallback_start_date is None:
            raise
        return [fallback_start_date + pd.Timedelta(days=i) for i in range(ntime)]


def rh_calc(T_celsius, pressure_hpa, specific_humidity):
    """
    Relative humidity from specific humidity.
    Same formula as your original bulk-flux script.
    """
    es = 6.1121 * np.exp(17.502 * T_celsius / (T_celsius + 240.97))
    es = es * (1.0007 + 3.46e-6 * pressure_hpa)
    em = specific_humidity * pressure_hpa / (0.378 * specific_humidity + 0.622)
    rh_val = 100.0 * em / es
    return np.clip(rh_val, 0.0, 100.0)


# ============================================================
# Runtime normalizer
# ============================================================

class RuntimeNorm:
    """
    Minimal dataset-like object for real forecast.

    It keeps the same normalization / denormalization logic as ForecastLoader,
    but does NOT load validation GLORYS truth or precomputed flux files.

    Required files under data_path:
        glosuf_GLORYS/glorys_deptho_mdt_122E155E_18N38N.nc
        glosuf_GLORYS/glorys_constant_mean.npy
        glosuf_GLORYS/glorys_constant_std.npy
        glosuf_GLORYS/glorys_ocean_mean.npy
        glosuf_GLORYS/glorys_ocean_std.npy
        glosuf_GLORYS/glorys_ocean_diff_mean.npy
        glosuf_GLORYS/glorys_ocean_diff_std.npy
        bulk_flux/bulk_flux_mean.npy
        bulk_flux/bulk_flux_std.npy
    """

    def __init__(self, data_path, device, expected_shape=(241, 397)):
        self.data_path = data_path
        self.device = device
        self.expected_shape = tuple(expected_shape)

        self._load_constants()
        self._load_normalization_factors()

    def p(self, *parts):
        return os.path.join(self.data_path, *parts)

    def to_tensor(self, arr):
        return torch.as_tensor(np.asarray(arr, dtype=np.float32), dtype=torch.float32, device=self.device)

    def _fix_constant_shape(self, arr, name):
        arr = as_np_float32(arr)

        if arr.shape == self.expected_shape:
            return arr

        if arr.T.shape == self.expected_shape:
            print(f"[Info] Transpose constant field {name}: {arr.shape} -> {arr.T.shape}")
            return arr.T.copy()

        raise ValueError(
            f"{name} shape mismatch: got {arr.shape}, "
            f"expected {self.expected_shape} or its transpose."
        )

    def _load_constants(self):
        const_file = self.p("glosuf_GLORYS", "glorys_deptho_mdt_122E155E_18N38N.nc")

        if not os.path.exists(const_file):
            raise FileNotFoundError(f"Constant file not found: {const_file}")

        with nc.Dataset(const_file, "r") as ds:
            if "deptho" not in ds.variables:
                raise KeyError(f"Variable deptho not found in {const_file}")
            if "mdt" not in ds.variables:
                raise KeyError(f"Variable mdt not found in {const_file}")

            deptho = self._fix_constant_shape(ds.variables["deptho"][:], "deptho")
            mdt = self._fix_constant_shape(ds.variables["mdt"][:], "mdt")

            if "ocean_mask" in ds.variables:
                mask = ds.variables["ocean_mask"][:]
                if np.ma.isMaskedArray(mask):
                    mask = mask.filled(0)
                mask = np.asarray(mask).astype(bool)

                if mask.shape != self.expected_shape:
                    if mask.T.shape == self.expected_shape:
                        mask = mask.T.copy()
                    else:
                        raise ValueError(
                            f"ocean_mask shape mismatch: got {mask.shape}, "
                            f"expected {self.expected_shape}."
                        )
            else:
                mask = np.isfinite(deptho)

        constant_mean = self.to_tensor(
            np.load(self.p("glosuf_GLORYS", "glorys_constant_mean.npy"))
        )
        constant_std = self.to_tensor(
            np.load(self.p("glosuf_GLORYS", "glorys_constant_std.npy"))
        )

        deptho_t = self.to_tensor(deptho)
        mdt_t = self.to_tensor(mdt)

        self.glorys_deptho = (deptho_t - constant_mean[0, 0]) / constant_std[0, 0]
        self.glorys_mdt = (mdt_t - constant_mean[1, 0]) / constant_std[1, 0]

        self.glorys_deptho = torch.nan_to_num(self.glorys_deptho, nan=0.0)
        self.glorys_mdt = torch.nan_to_num(self.glorys_mdt, nan=0.0)

        self.ocean_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        if self.glorys_deptho.shape != self.expected_shape:
            raise ValueError(f"Normalized deptho shape error: {self.glorys_deptho.shape}")

        if self.glorys_mdt.shape != self.expected_shape:
            raise ValueError(f"Normalized mdt shape error: {self.glorys_mdt.shape}")

        if self.ocean_mask.shape != self.expected_shape:
            raise ValueError(f"ocean_mask shape error: {self.ocean_mask.shape}")

    def _load_normalization_factors(self):
        self.glorys_means = self.to_tensor(
            np.load(self.p("glosuf_GLORYS", "glorys_ocean_mean.npy"))
        )
        self.glorys_stds = self.to_tensor(
            np.load(self.p("glosuf_GLORYS", "glorys_ocean_std.npy"))
        )

        self.flux_means = self.to_tensor(
            np.load(self.p("bulk_flux", "bulk_flux_mean.npy"))
        )
        self.flux_stds = self.to_tensor(
            np.load(self.p("bulk_flux", "bulk_flux_std.npy"))
        )

        self.glorys_diff_means = self.to_tensor(
            np.load(self.p("glosuf_GLORYS", "glorys_ocean_diff_mean.npy"))
        )
        self.glorys_diff_stds = self.to_tensor(
            np.load(self.p("glosuf_GLORYS", "glorys_ocean_diff_std.npy"))
        )

        if self.glorys_means.shape[0] != 5:
            raise ValueError(f"glorys_ocean_mean shape error: {self.glorys_means.shape}")

        if self.flux_means.shape[0] != 8:
            raise ValueError(f"bulk_flux_mean shape error: {self.flux_means.shape}")

        if self.glorys_diff_means.shape[0] != 5:
            raise ValueError(f"glorys_ocean_diff_mean shape error: {self.glorys_diff_means.shape}")

    def norm_glorys(self, x, var_name):
        idx = VAR_NAMES.index(var_name)
        mean = self.glorys_means[idx, 0].view(1, 1)
        std = self.glorys_stds[idx, 0].view(1, 1)
        return (x - mean) / std

    def norm_era5fluxes(self, x):
        means = self.flux_means[:, 0].view(8, 1, 1)
        stds = self.flux_stds[:, 0].view(8, 1, 1)
        return (x - means) / stds

    def unnorm_tendency(self, preds):
        """
        preds can be:
            [B, lon, lat, 5]
            [B, lat, lon, 5]
            [B, 5, lat, lon]
            [B, 5, lon, lat]
        """
        means = self.glorys_diff_means[:, 0]
        stds = self.glorys_diff_stds[:, 0]

        if preds.ndim != 4:
            raise ValueError(f"Unexpected preds ndim: {preds.shape}")

        if preds.shape[-1] == 5:
            means = means.view(1, 1, 1, 5).to(preds.device)
            stds = stds.view(1, 1, 1, 5).to(preds.device)
        elif preds.shape[1] == 5:
            means = means.view(1, 5, 1, 1).to(preds.device)
            stds = stds.view(1, 5, 1, 1).to(preds.device)
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

        return preds * stds + means

    def make_time_channels_by_date(self, date, x, y):
        """
        Same logic as ForecastLoader.make_time_channels,
        but using actual pandas Timestamp instead of validation-set index.
        """
        ts = pd.Timestamp(date)
        doy = ts.dayofyear - 1
        n_days = 366 if ts.is_leap_year else 365

        one = np.ones((1, x, y), dtype=np.float32)
        doy_sin = np.sin(doy * 2 * np.pi / n_days).astype(np.float32) * one
        doy_cos = np.cos(doy * 2 * np.pi / n_days).astype(np.float32) * one

        return np.concatenate([doy_sin, doy_cos], axis=0)


# ============================================================
# Reader for actual GLO12v4 + ERA5 combined nc file
# ============================================================

class GLO12v4ERA5Forcing:
    """
    Reader for:
        GLO12v4_ERA5_YYYYMMDD_12days_on_target_grid.nc

    Input nc variable layout from MATLAB:
        longitude: [397]
        latitude : [241]
        time     : [12]

        uo/vo/thetao/so: [longitude, latitude, depth, time]
        zos            : [longitude, latitude, time]

        ERA5 variables: [longitude, latitude, time]

    This class returns:
        ocean state: [5, 241, 397] = [var, lat, lon]
        atmosphere : dict, each [241, 397] = [lat, lon]
    """

    def __init__(self, input_nc):
        self.input_nc = input_nc

        if not os.path.exists(input_nc):
            raise FileNotFoundError(f"Input nc not found: {input_nc}")

        self.ds = nc.Dataset(input_nc, "r")

        self._validate()
        self._load_coordinates_and_dates()

    def _validate(self):
        required = ["longitude", "latitude", "time"] + VAR_NAMES + ATM_NAMES
        missing = [v for v in required if v not in self.ds.variables]

        if missing:
            raise KeyError(f"Missing variables in input nc: {missing}")

        if "depth" not in self.ds.dimensions and "depth" not in self.ds.variables:
            raise KeyError("Missing depth dimension / variable.")

    def _load_coordinates_and_dates(self):
        self.lon = as_np_float32(self.ds.variables["longitude"][:]).reshape(-1)
        self.lat = as_np_float32(self.ds.variables["latitude"][:]).reshape(-1)

        self.nlon = len(self.lon)
        self.nlat = len(self.lat)
        self.shape = (self.nlat, self.nlon)

        self.ntime = len(self.ds.variables["time"][:])

        start_from_name = parse_start_date_from_filename(self.input_nc)
        self.dates = nc_time_to_pandas_dates(self.ds, fallback_start_date=start_from_name)

        if len(self.dates) != self.ntime:
            raise ValueError("Decoded dates length mismatch with time dimension.")

        if self.ntime < 12:
            raise ValueError(f"Input nc has only {self.ntime} time steps. Need at least 12.")

        self.lat_grid = np.repeat(self.lat[:, None], self.nlon, axis=1).astype(np.float64)

    def _read_var_2d(self, var_name, time_index, depth_index=0):
        """
        Read one variable at one time as [lat, lon].
        """
        if var_name not in self.ds.variables:
            raise KeyError(f"{var_name} not found in {self.input_nc}")

        v = self.ds.variables[var_name]
        dims = tuple(v.dimensions)
        dims_low = tuple(d.lower() for d in dims)

        slices = []
        remaining_dims = []

        for dim_name, dim_low in zip(dims, dims_low):
            if dim_low == "time":
                slices.append(int(time_index))
            elif dim_low in ["depth", "deptht", "depthu", "depthv", "lev"]:
                slices.append(int(depth_index))
            else:
                slices.append(slice(None))
                remaining_dims.append(dim_low)

        arr = v[tuple(slices)]
        arr = as_np_float32(arr)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(
                f"{var_name} after slicing should be 2D, got shape {arr.shape}. "
                f"Original dimensions: {dims}"
            )

        if len(remaining_dims) != 2:
            raise ValueError(
                f"{var_name} remaining spatial dims error: {remaining_dims}. "
                f"Original dimensions: {dims}"
            )

        # Convert to [lat, lon].
        if remaining_dims == ["latitude", "longitude"] or remaining_dims == ["lat", "lon"]:
            out = arr
        elif remaining_dims == ["longitude", "latitude"] or remaining_dims == ["lon", "lat"]:
            out = arr.T
        else:
            raise ValueError(
                f"Unsupported spatial dimension order for {var_name}: {remaining_dims}"
            )

        if out.shape != self.shape:
            raise ValueError(
                f"{var_name} shape error after conversion: {out.shape}, "
                f"expected {self.shape} = [lat, lon]."
            )

        return out.astype(np.float32)

    def get_ocean(self, time_index):
        """
        Return ocean state [5, lat, lon]:
            uo, vo, thetao, so, zos
        """
        uo = self._read_var_2d("uo", time_index, depth_index=0)
        vo = self._read_var_2d("vo", time_index, depth_index=0)
        thetao = self._read_var_2d("thetao", time_index, depth_index=0)
        so = self._read_var_2d("so", time_index, depth_index=0)
        zos = self._read_var_2d("zos", time_index, depth_index=0)

        ocean = np.stack([uo, vo, thetao, so, zos], axis=0).astype(np.float32)

        if ocean.shape != (5, self.nlat, self.nlon):
            raise ValueError(f"Ocean state shape error: {ocean.shape}")

        return ocean

    def get_day(self, time_index):
        """
        Return ERA5 atmospheric forcing dict.
        Each variable is [lat, lon].
        """
        atm = {}
        for name in ATM_NAMES:
            atm[name] = self._read_var_2d(name, time_index, depth_index=0)

        return atm

    def close(self):
        if getattr(self, "ds", None) is not None:
            self.ds.close()
            self.ds = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# ============================================================
# Online COARE bulk flux
# ============================================================

def compute_bulk_flux_np(
    ocean_raw_np,
    atm,
    lat_grid,
    ocean_mask_np,
    pressure_hpa=1015.0,
):
    """
    Compute bulk flux using predicted ocean state and one ERA5 day.

    Input:
        ocean_raw_np: [5, 241, 397], unnormalized ocean state
                      order = uo, vo, thetao, so, zos
        atm: dict containing u10, v10, t2m, sh, ssrd, strd, tp
             each [241, 397]
        lat_grid: [241, 397]
        ocean_mask_np: [241, 397] bool

    Output:
        flux_np: [8, 241, 397]
                 order = tau_u, tau_v, sensible, latent,
                         net_short, net_long, evap, rain
    """
    if ocean_raw_np.ndim != 3 or ocean_raw_np.shape[0] != 5:
        raise ValueError(f"Unexpected ocean_raw_np shape: {ocean_raw_np.shape}")

    sea_u = ocean_raw_np[0].astype(np.float64)
    sea_v = ocean_raw_np[1].astype(np.float64)
    sst = ocean_raw_np[2].astype(np.float64)

    u10 = atm["u10"].astype(np.float64)
    v10 = atm["v10"].astype(np.float64)
    t2m_c = atm["t2m"].astype(np.float64) - 273.15
    sh = atm["sh"].astype(np.float64)
    ssrd = atm["ssrd"].astype(np.float64)
    strd = atm["strd"].astype(np.float64)
    tp = atm["tp"].astype(np.float64)

    # Keep same convention as your previous code:
    # current velocity is not subtracted.
    du = u10 - sea_u
    dv = v10 - sea_v
    spd_mag = np.sqrt(du ** 2 + dv ** 2)

    rh = rh_calc(t2m_c, pressure_hpa, sh)

    # Same conversion as your previous bulk-flux code.
    Rs = ssrd / 3600.0
    Rl = strd / 3600.0
    rain_rate = tp * 1000.0

    valid = (
        ocean_mask_np
        & np.isfinite(spd_mag)
        & np.isfinite(t2m_c)
        & np.isfinite(rh)
        & np.isfinite(sst)
        & np.isfinite(Rs)
        & np.isfinite(Rl)
        & np.isfinite(rain_rate)
    )

    nlat, nlon = ocean_mask_np.shape

    tau = np.full((nlat, nlon), np.nan, dtype=np.float64)
    hsb = np.full((nlat, nlon), np.nan, dtype=np.float64)
    hlb = np.full((nlat, nlon), np.nan, dtype=np.float64)
    rsn = np.full((nlat, nlon), np.nan, dtype=np.float64)
    rln = np.full((nlat, nlon), np.nan, dtype=np.float64)
    evap = np.full((nlat, nlon), np.nan, dtype=np.float64)
    rain_out = np.full((nlat, nlon), np.nan, dtype=np.float64)

    if np.any(valid):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            A_valid = coare35vn.coare35vn(
                u=spd_mag[valid],
                t=t2m_c[valid],
                rh=rh[valid],
                ts=sst[valid],
                Rs=Rs[valid],
                Rl=Rl[valid],
                rain=rain_rate[valid],
                P=pressure_hpa,
                zu=10,
                zt=2,
                zq=2,
                lat=lat_grid[valid],
                zi=600,
            )

        tau[valid] = A_valid[:, 0]
        hsb[valid] = A_valid[:, 1]
        hlb[valid] = A_valid[:, 2]
        rsn[valid] = A_valid[:, 3]
        rln[valid] = A_valid[:, 4]
        evap[valid] = A_valid[:, 5]
        rain_out[valid] = A_valid[:, 6]

    with np.errstate(divide="ignore", invalid="ignore"):
        tau_u = tau * du / spd_mag
        tau_v = tau * dv / spd_mag

    flux_np = np.stack(
        [tau_u, tau_v, hsb, hlb, rsn, rln, evap, rain_out],
        axis=0,
    ).astype(np.float32)

    flux_np[:, ~ocean_mask_np] = np.nan
    flux_np[~np.isfinite(flux_np)] = np.nan

    return flux_np


def compute_live_norm_flux(
    runtime,
    forcing,
    ocean_raw,
    atm_index,
    ocean_mask_np,
    device,
    pressure_hpa=1015.0,
):
    """
    Online bulk flux computation + normalization.

    Input:
        ocean_raw: torch tensor [5, 241, 397]
        atm_index: index in input GLO12v4_ERA5 nc file

    Output:
        flux_norm: torch tensor [8, 241, 397]
    """
    ocean_raw_np = ocean_raw.detach().cpu().numpy().astype(np.float32)
    atm = forcing.get_day(atm_index)

    flux_np = compute_bulk_flux_np(
        ocean_raw_np=ocean_raw_np,
        atm=atm,
        lat_grid=forcing.lat_grid,
        ocean_mask_np=ocean_mask_np,
        pressure_hpa=pressure_hpa,
    )

    flux = to_torch(flux_np, device)
    flux = runtime.norm_era5fluxes(flux)
    flux = torch.nan_to_num(flux, nan=0.0)

    return flux


# ============================================================
# Model loading and task construction
# ============================================================

def load_trained_model(
    checkpoint_path,
    device,
    in_channels=38,
    out_channels=5,
    int_channels=256,
):
    model = ConvCNPSCS(
        in_channels=in_channels,
        out_channels=out_channels,
        int_channels=int_channels,
        device=device,
    ).to(device)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def norm_surface(runtime, surf_raw):
    """
    Normalize ocean state.

    Input:
        surf_raw: [5, 241, 397]
    Output:
        surf_norm: [5, 241, 397]
    """
    uo = runtime.norm_glorys(surf_raw[0], "uo")
    vo = runtime.norm_glorys(surf_raw[1], "vo")
    thetao = runtime.norm_glorys(surf_raw[2], "thetao")
    so = runtime.norm_glorys(surf_raw[3], "so")
    zos = runtime.norm_glorys(surf_raw[4], "zos")

    surf_norm = torch.stack([uo, vo, thetao, so, zos], dim=0)
    surf_norm = torch.nan_to_num(surf_norm, nan=0.0)

    return surf_norm


def build_task_real_livebulk(
    runtime,
    forcing,
    idx_prev,
    idx_curr,
    idx_target,
    ocean_prev_raw,
    ocean_curr_raw,
    ocean_mask_np,
    device,
    pressure_hpa=1015.0,
):
    """
    Build model input for one autoregressive step.

    Channel order:
        F_{t-1}, O_{t-1},
        F_t,     O_t,
        F_{t+1|t},
        deptho, mdt,
        time

    Shape:
        y_context: [1, 38, 241, 397]
    """
    flux_prev = compute_live_norm_flux(
        runtime=runtime,
        forcing=forcing,
        ocean_raw=ocean_prev_raw,
        atm_index=idx_prev,
        ocean_mask_np=ocean_mask_np,
        device=device,
        pressure_hpa=pressure_hpa,
    )

    flux_curr = compute_live_norm_flux(
        runtime=runtime,
        forcing=forcing,
        ocean_raw=ocean_curr_raw,
        atm_index=idx_curr,
        ocean_mask_np=ocean_mask_np,
        device=device,
        pressure_hpa=pressure_hpa,
    )

    flux_futu = compute_live_norm_flux(
        runtime=runtime,
        forcing=forcing,
        ocean_raw=ocean_curr_raw,
        atm_index=idx_target,
        ocean_mask_np=ocean_mask_np,
        device=device,
        pressure_hpa=pressure_hpa,
    )

    ocean_prev_norm = norm_surface(runtime, ocean_prev_raw)
    ocean_curr_norm = norm_surface(runtime, ocean_curr_raw)

    target_date = forcing.dates[idx_target]

    time = to_torch(
        runtime.make_time_channels_by_date(
            target_date,
            ocean_curr_raw.shape[1],
            ocean_curr_raw.shape[2],
        ),
        device,
    )

    y_context = torch.cat(
        [
            flux_prev,
            ocean_prev_norm,
            flux_curr,
            ocean_curr_norm,
            flux_futu,
            runtime.glorys_deptho.unsqueeze(0),
            runtime.glorys_mdt.unsqueeze(0),
            time,
        ],
        dim=0,
    )

    expected_shape = (38, forcing.nlat, forcing.nlon)

    if tuple(y_context.shape) != expected_shape:
        raise ValueError(
            f"Expected y_context shape {expected_shape}, got {tuple(y_context.shape)}"
        )

    return {
        "y_context": y_context.unsqueeze(0).to(device),
        "lt": torch.tensor([[1.0]], dtype=torch.float32, device=device),
        "target_index": torch.tensor([idx_target], dtype=torch.long, device=device),
    }


def denorm_tendency_to_ocean_layout(runtime, pred_norm_tendency, nlat=241, nlon=397):
    """
    Convert model output tendency to [5, lat, lon].

    Supports:
        [B, lon, lat, 5]
        [B, lat, lon, 5]
        [B, 5, lat, lon]
        [B, 5, lon, lat]
    """
    pred_raw = runtime.unnorm_tendency(pred_norm_tendency)

    if pred_raw.ndim != 4:
        raise ValueError(f"Unexpected model output ndim: {pred_raw.shape}")

    # Case A: [B, lon, lat, 5]
    if pred_raw.shape[-1] == 5 and pred_raw.shape[1] == nlon and pred_raw.shape[2] == nlat:
        return pred_raw[0].permute(2, 1, 0).contiguous()

    # Case B: [B, lat, lon, 5]
    if pred_raw.shape[-1] == 5 and pred_raw.shape[1] == nlat and pred_raw.shape[2] == nlon:
        return pred_raw[0].permute(2, 0, 1).contiguous()

    # Case C: [B, 5, lat, lon]
    if pred_raw.shape[1] == 5 and pred_raw.shape[2] == nlat and pred_raw.shape[3] == nlon:
        return pred_raw[0].contiguous()

    # Case D: [B, 5, lon, lat]
    if pred_raw.shape[1] == 5 and pred_raw.shape[2] == nlon and pred_raw.shape[3] == nlat:
        return pred_raw[0].permute(0, 2, 1).contiguous()

    raise ValueError(f"Unsupported model output shape: {pred_raw.shape}")


# ============================================================
# Save forecast output
# ============================================================

def save_forecast_to_nc(
    output_nc,
    input_nc,
    lon,
    lat,
    pred_array,
    valid_dates,
    init_prev_date,
    init_curr_date,
    depth_value=0.0,
):
    """
    Save predicted ocean states to NetCDF.

    Input:
        pred_array: [lead_days, 5, lat, lon]
                    variable order = uo, vo, thetao, so, zos

    Output variable layout:
        uo/vo/thetao/so: [longitude, latitude, depth, time]
        zos            : [longitude, latitude, time]
    """
    ensure_dir(os.path.dirname(output_nc))

    if os.path.exists(output_nc):
        os.remove(output_nc)

    pred_array = np.asarray(pred_array, dtype=np.float32)

    lead_days, nvar, nlat, nlon = pred_array.shape

    if nvar != 5:
        raise ValueError(f"pred_array variable dimension must be 5, got {nvar}")

    if nlat != len(lat) or nlon != len(lon):
        raise ValueError(
            f"pred_array spatial shape mismatch: pred {pred_array.shape}, "
            f"lon {len(lon)}, lat {len(lat)}"
        )

    time_units = "days since 1970-01-01 00:00:00"
    calendar = "proleptic_gregorian"

    py_dates = []
    for d in valid_dates:
        ts = pd.Timestamp(d)
        py_dates.append(dt.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second))

    time_values = nc.date2num(py_dates, units=time_units, calendar=calendar)

    with nc.Dataset(output_nc, "w", format="NETCDF4") as ds:
        # Dimensions
        ds.createDimension("longitude", nlon)
        ds.createDimension("latitude", nlat)
        ds.createDimension("depth", 1)
        ds.createDimension("time", lead_days)

        # Coordinates
        vlon = ds.createVariable("longitude", "f4", ("longitude",))
        vlat = ds.createVariable("latitude", "f4", ("latitude",))
        vdep = ds.createVariable("depth", "f4", ("depth",))
        vtim = ds.createVariable("time", "f8", ("time",))

        vlon[:] = lon.astype(np.float32)
        vlat[:] = lat.astype(np.float32)
        vdep[:] = np.asarray([depth_value], dtype=np.float32)
        vtim[:] = np.asarray(time_values, dtype=np.float64)

        vlon.standard_name = "longitude"
        vlon.long_name = "longitude"
        vlon.units = "degrees_east"

        vlat.standard_name = "latitude"
        vlat.long_name = "latitude"
        vlat.units = "degrees_north"

        vdep.standard_name = "depth"
        vdep.long_name = "surface depth"
        vdep.units = "m"
        vdep.positive = "down"

        vtim.standard_name = "time"
        vtim.long_name = "forecast valid time"
        vtim.units = time_units
        vtim.calendar = calendar

        fill_value = np.float32(np.nan)

        def create_4d_var(name, long_name, units):
            v = ds.createVariable(
                name,
                "f4",
                ("longitude", "latitude", "depth", "time"),
                zlib=True,
                complevel=4,
                fill_value=fill_value,
            )
            v.long_name = long_name
            v.units = units
            v.coordinates = "longitude latitude depth time"
            return v

        def create_3d_var(name, long_name, units):
            v = ds.createVariable(
                name,
                "f4",
                ("longitude", "latitude", "time"),
                zlib=True,
                complevel=4,
                fill_value=fill_value,
            )
            v.long_name = long_name
            v.units = units
            v.coordinates = "longitude latitude time"
            return v

        var_uo = create_4d_var("uo", "predicted eastward sea water velocity", "m s-1")
        var_vo = create_4d_var("vo", "predicted northward sea water velocity", "m s-1")
        var_thetao = create_4d_var("thetao", "predicted sea water potential temperature", "degree_C")
        var_so = create_4d_var("so", "predicted sea water salinity", "1e-3")
        var_zos = create_3d_var("zos", "predicted sea surface height above geoid", "m")

        # Write data.
        # pred_array: [time, var, lat, lon]
        # output 4D: [lon, lat, depth, time]
        var_uo[:, :, 0, :] = np.transpose(pred_array[:, 0, :, :], (2, 1, 0))
        var_vo[:, :, 0, :] = np.transpose(pred_array[:, 1, :, :], (2, 1, 0))
        var_thetao[:, :, 0, :] = np.transpose(pred_array[:, 2, :, :], (2, 1, 0))
        var_so[:, :, 0, :] = np.transpose(pred_array[:, 3, :, :], (2, 1, 0))

        # output 3D: [lon, lat, time]
        var_zos[:, :, :] = np.transpose(pred_array[:, 4, :, :], (2, 1, 0))

        # Global attributes
        ds.title = "10-day autoregressive ocean forecast from GLO12v4 and ERA5 forcing"
        ds.source_input_file = input_nc
        ds.init_prev_date = str(pd.Timestamp(init_prev_date).date())
        ds.init_curr_date = str(pd.Timestamp(init_curr_date).date())
        ds.first_forecast_valid_date = str(pd.Timestamp(valid_dates[0]).date())
        ds.last_forecast_valid_date = str(pd.Timestamp(valid_dates[-1]).date())
        ds.lead_days = int(lead_days)
        ds.variable_order = "uo, vo, thetao, so, zos"
        ds.flux_mode = "online_bulk_from_predicted_ocean_and_ERA5"
        ds.data_precision = "single"
        ds.history = "Created by forecast_real_from_GLO12v4_ERA5_nc.py"


# ============================================================
# Main real forecast function
# ============================================================

@torch.no_grad()
def forecast_real_from_combined_nc(
    input_nc,
    checkpoint_path,
    output_nc,
    data_path="./high_resolution_data/",
    lead_days=10,
    device="cuda",
    in_channels=38,
    out_channels=5,
    int_channels=256,
    pressure_hpa=1015.0,
):
    """
    Real application forecast from one combined GLO12v4 + ERA5 nc file.

    Required input:
        time index 0: O(t-1), A(t-1)
        time index 1: O(t),   A(t)
        time index 2..11: ERA5 forcing A(t+1)..A(t+10)

    Output:
        forecast valid days:
            input time index 2..11
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Real GLO12v4 + ERA5 autoregressive forecast")
    print(f"Input nc        : {input_nc}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Output nc       : {output_nc}")
    print(f"Data path       : {data_path}")
    print(f"Lead days       : {lead_days}")
    print(f"Device          : {device}")
    print("=" * 80)

    with GLO12v4ERA5Forcing(input_nc) as forcing:
        if lead_days + 2 > forcing.ntime:
            raise ValueError(
                f"Input nc has {forcing.ntime} time steps. "
                f"Need at least lead_days + 2 = {lead_days + 2}."
            )

        if forcing.shape != (241, 397):
            raise ValueError(
                f"This model code expects [lat, lon] = [241, 397], "
                f"but input file has {forcing.shape}."
            )

        runtime = RuntimeNorm(
            data_path=data_path,
            device=device,
            expected_shape=forcing.shape,
        )

        model = load_trained_model(
            checkpoint_path=checkpoint_path,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            int_channels=int_channels,
        )

        ocean_mask = runtime.ocean_mask
        ocean_mask_np = ocean_mask.detach().cpu().numpy().astype(bool)

        # Initial fields:
        # day 0 = O(t-1)
        # day 1 = O(t)
        idx_prev = 0
        idx_curr = 1

        ocean_prev_raw = to_torch(forcing.get_ocean(idx_prev), device)
        ocean_curr_raw = to_torch(forcing.get_ocean(idx_curr), device)

        ocean_prev_raw[:, ~ocean_mask] = torch.nan
        ocean_curr_raw[:, ~ocean_mask] = torch.nan

        print(f"Input dates:")
        for i in range(min(forcing.ntime, 12)):
            print(f"  index {i:02d}: {pd.Timestamp(forcing.dates[i]).date()}")

        print("\nInitial condition:")
        print(f"  O(t-1) index {idx_prev}: {pd.Timestamp(forcing.dates[idx_prev]).date()}")
        print(f"  O(t)   index {idx_curr}: {pd.Timestamp(forcing.dates[idx_curr]).date()}")

        pred_list = []
        valid_dates = []

        for lead in range(1, lead_days + 1):
            idx_target = idx_curr + 1

            if idx_target >= forcing.ntime:
                raise IndexError(
                    f"Need atmospheric forcing at index {idx_target}, "
                    f"but input file has only {forcing.ntime} time steps."
                )

            target_date = forcing.dates[idx_target]

            print(
                f"\nLead {lead:02d}/{lead_days}: "
                f"predicting {pd.Timestamp(target_date).date()} "
                f"using A index {idx_target}"
            )

            task = build_task_real_livebulk(
                runtime=runtime,
                forcing=forcing,
                idx_prev=idx_prev,
                idx_curr=idx_curr,
                idx_target=idx_target,
                ocean_prev_raw=ocean_prev_raw,
                ocean_curr_raw=ocean_curr_raw,
                ocean_mask_np=ocean_mask_np,
                device=device,
                pressure_hpa=pressure_hpa,
            )

            pred_norm_tendency = model(task, film_index=1.0)

            pred_raw_tendency = denorm_tendency_to_ocean_layout(
                runtime=runtime,
                pred_norm_tendency=pred_norm_tendency,
                nlat=forcing.nlat,
                nlon=forcing.nlon,
            )

            # Autoregressive update:
            # O_pred(t+1) = O_pred(t) + ΔO_pred(t+1)
            ocean_next_raw = ocean_curr_raw + pred_raw_tendency
            ocean_next_raw[:, ~ocean_mask] = torch.nan

            pred_np = ocean_next_raw.detach().cpu().numpy().astype(np.float32)
            pred_list.append(pred_np)
            valid_dates.append(pd.Timestamp(target_date))

            finite_counts = [
                int(np.isfinite(pred_np[i]).sum()) for i in range(len(VAR_NAMES))
            ]
            print("  finite counts:", dict(zip(VAR_NAMES, finite_counts)))

            # Slide window:
            # previous <- current
            # current  <- predicted next
            ocean_prev_raw = ocean_curr_raw.detach()
            ocean_curr_raw = ocean_next_raw.detach()

            idx_prev = idx_curr
            idx_curr = idx_target

        pred_array = np.stack(pred_list, axis=0).astype(np.float32)
        # pred_array: [lead_days, 5, lat, lon]

        print("\nPrediction array shape:", pred_array.shape)

        # Try to keep input depth value if available.
        try:
            depth_value = float(np.asarray(forcing.ds.variables["depth"][:]).reshape(-1)[0])
        except Exception:
            depth_value = 0.0

        save_forecast_to_nc(
            output_nc=output_nc,
            input_nc=input_nc,
            lon=forcing.lon,
            lat=forcing.lat,
            pred_array=pred_array,
            valid_dates=valid_dates,
            init_prev_date=forcing.dates[0],
            init_curr_date=forcing.dates[1],
            depth_value=depth_value,
        )

    print("\nForecast completed.")
    print(f"Saved to: {output_nc}")

    return {
        "output_nc": output_nc,
        "valid_dates": [str(pd.Timestamp(d).date()) for d in valid_dates],
        "variables": VAR_NAMES,
        "pred_shape": pred_array.shape,
    }


# ============================================================
# Script entry
# ============================================================

if __name__ == "__main__":

    # 修改为你的实际路径
    input_nc = r"/content/drive/MyDrive/OM_major_revision/high_resolution_study/high_resolution_data/GLO12v4/GLO12v4_ERA5_20260412_12days_on_target_grid.nc"

    checkpoint_path = r"/content/drive/MyDrive/OM_major_revision/high_resolution_study/model_train_withfutureflux_xitai/epoch_126"

    output_nc = r"/content/drive/MyDrive/OM_major_revision/high_resolution_study/GLO12v4_output/forecast_model_output_20260412_10days.nc"

    # 这里必须指向包含归一化 npy 和常量 nc 的目录
    # 即原 ForecastLoader 里的 self.data_path
    data_path = r"./high_resolution_data/"

    forecast_real_from_combined_nc(
        input_nc=input_nc,
        checkpoint_path=checkpoint_path,
        output_nc=output_nc,
        data_path=data_path,
        lead_days=10,
        device="cuda",
        in_channels=38,
        out_channels=5,
        int_channels=256,
        pressure_hpa=1015.0,
    )