import netCDF4 as nc

data_path = "E:/OM_revision_data/1deg_ablation_data/"
with nc.Dataset(data_path + "bulk_flux/Calculated_Flux_2000_1deg.nc",
                "r") as glorys_file:
    glorys_u = glorys_file.variables["tau_u"][:]
    glorys_v = glorys_file.variables["tau_v"][:]
    glorys_sens = glorys_file.variables["sensible"][:]
    glorys_late = glorys_file.variables["latent"][:]
    glorys_ns = glorys_file.variables["net_short"][:]
    glorys_nl = glorys_file.variables["net_long"][:]
    glorys_evap = glorys_file.variables["evap"][:]
    glorys_rain = glorys_file.variables["rain"][:]
print(glorys_u.shape)