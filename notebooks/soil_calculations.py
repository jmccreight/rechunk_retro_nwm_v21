import numpy as np
import xarray as xr

file_soil_param = (
    "/glade/p/cisl/nwc/nwmv21_finals/CONUS/DOMAIN_FinalParams/"
    "soil_properties_FullRouting_NWMv2.1.nc")
ds_soil_param = (
    xr.open_dataset(file_soil_param)
    .squeeze('Time'))
ds_soil_param = ds_soil_param.rename_dims(
    {'west_east': 'x', 'south_north': 'y'})

da_soil_depth = xr.DataArray(
    #  meters:        layer depth            
    data=np.array([0.1, 0.3, 0.6, 1.0]),  # hard-coded for NWM v2.1
    dims=["soil_layers_stag"],
    coords=dict(soil_layers_stag=ds_soil_param.soil_layers_stag),
    attrs=dict(description="soil depth", units="m"))

da_soil_depth_frac = xr.DataArray(
    #  fraction:        layer depth / total depth
    data=da_soil_depth.values / da_soil_depth.sum().values,
    dims=["soil_layers_stag"],
    coords=dict(soil_layers_stag=ds_soil_param.soil_layers_stag),
    attrs=dict(description="soil depth fraction", units="-"))

da_soil_volume = xr.DataArray(
    #  m^3:     layer depth   *  m   *  m         
    data=da_soil_depth.values * 1000 * 1000,  # hard-coded for NWM v2.1
    dims=["soil_layers_stag"],
    coords=dict(soil_layers_stag=ds_soil_param.soil_layers_stag),
    attrs=dict(description="soil volume", units="m^3"))


def soil_water_volume(ds_ldasout):
    return (
        (ds_ldasout.SOIL_M * da_soil_volume)
        .sum(dim='soil_layers_stag')
        .rename('soil_water_volume'))


def soil_water_pct_sat(ds_ldasout):
    return (
        ((ds_ldasout.SOIL_M / ds_soil_param.smcmax) * da_soil_depth_frac)
        .sum(dim='soil_layers_stag')
        .rename('soil_water_pct_sat'))

