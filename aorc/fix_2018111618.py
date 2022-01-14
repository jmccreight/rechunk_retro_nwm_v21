import numpy as np
import pathlib
import xarray as xr

file_bad = pathlib.Path(
    '/glade/p/cisl/nwc/nwm_forcings/AORC/2018111618.LDASIN_DOMAIN1')
ds_bad = xr.open_dataset(file_bad)

ds_fix = ds_bad.drop('valid_time').copy()

ds_fix['valid_time'] = xr.DataArray(
    np.array(['2018-11-16T18:00:00.000000000'], dtype='datetime64[ns]'),
    dims=['Time'])

file_replacement = pathlib.Path(
    '/glade/scratch/jamesmcc/aorc_forcing_symlinks/'
    '2018/201811161800.LDASIN_DOMAIN1')

if file_replacement.exists():
    file_replacement.unlink()

ds_fix.to_netcdf(file_replacement)
ds_fix.close()

ds_check = xr.open_dataset(file_replacement)
del ds_check


def preprocess_precip(ds):
    drop_vars_full = [
        "reference_time", "crs", 'U2D',
        'V2D', 'LWDOWN', 'T2D',
        'Q2D', 'PSFC', 'SWDOWN',
        'LQFRAC', 'x', 'y']
    drop_vars = list(
        set(ds.variables).intersection(set(drop_vars_full)))
    ds = ds.drop(drop_vars)
    if 'valid_time' in ds.variables:
        ds= (ds
             .rename({'valid_time': 'time'})
             .set_coords('time')
             .swap_dims({'Time': 'time'})
             .drop('Times')
             .rename({'south_north': 'y', 'west_east': 'x'}))
    return ds.reset_coords(drop=True)


m_files = sorted(file_replacement.parent.glob('201811161*'))
ds_check_2 = xr.open_mfdataset(
    m_files,
    parallel=True,
    preprocess=preprocess_precip,
    combine="by_coords",
    concat_dim="time",
    join="override",)
