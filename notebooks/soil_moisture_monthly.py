# Soil water to monthly averages for the retrospective
# This calculate the total soil water in the noahMP column
# and takes a monthly average.

# Notes:
#   This script takes a single argument assumed to be in 0-4.
#   This is the "team id" and this member/team_id will process
#   spatial chunks equal to chunk % 4 = team_id


# Using virtual environment /glade/work/jamesmcc/python_envs/712gc
import os

if 'DASK_ROOT_CONFIG' in os.environ:
    del os.environ['DASK_ROOT_CONFIG']

import dask
from dask.distributed import Client, progress, LocalCluster, performance_report
from dask_jobqueue import PBSCluster

import datetime
import math
import numpy as np
import pathlib
import socket
import sys
import time
import xarray as xr

# from soil_calcualations import soil_water_volume, soil_water_pct_sat
import soil_calculations

n_teams = 1
n_workers = 16
n_workers_start = 8
variable = 'soil_water_volume'
the_soil_calculation = getattr(soil_calculations, variable)
dir_proj = pathlib.Path('/glade/scratch/jamesmcc/soil_moisture_monthly/')
file_output = dir_proj / f'{variable}.zarr'
file_ldasout = '/glade/p/datashare/ishitas/nwm_retro_v2.1/ldasout.zarr'
file_comp_log = dir_proj / f'{variable}_complete.log'


def read_log_to_list(file_log):
    with file_log.open("r") as opened_file:
        line_list = [line.strip() for line in opened_file]
    return line_list


def append_key_to_log(key, file_log):
    with file_log.open('a') as opened_file:
        opened_file.write(f'{key}\n')
    return None


if __name__ == "__main__":

    team_id = 0  # int(sys.argv[-1])
    print(f'\nteam_id: {team_id}', flush=True)

    if (team_id == 0) and (not file_comp_log.exists()):
        file_comp_log.touch()

    cluster = PBSCluster(
        cores=1,
        queue='casper', 
        project='NRAL0017', 
        memory='80GB', 
        walltime='07:00:00',
        death_timeout=75)

    cluster.scale(jobs=n_workers)
    client = Client(cluster)
    print("Waiting for workers...", flush=True)
    client.wait_for_workers(n_workers)
    print("have workers!", flush=True)

    dash_link = client.dashboard_link
    print(dash_link, flush=True)
    hostname = socket.gethostname()
    user = os.environ["USER"]
    port = dash_link.split(':')[-1].split('/')[0]
    print(f"ssh -NL {port}:{hostname}:{port} {user}@casper.ucar.edu", flush=True)
    print(client, flush=True)

    # Soil water calculations
    ds_ldasout = xr.open_zarr(file_ldasout)
    da_soil_calculation =the_soil_calculation(ds_ldasout)
    da_soil_calc_monthly = (
        da_soil_calculation
        .resample(time="1M").mean()
        .chunk({'time': 224, 'y': 350, 'x': 350}))

    if not file_output.exists():
        if team_id == 0:
            da_soil_calc_monthly.to_dataset().to_zarr(
                file_output, compute=False, consolidated=True)
            # This took 4min for the full dataset
        else:
            raise FileExistsError(f'No such file: {file_output}')
            # Could do a "lock" file here or just run a single scipt
            # once off-line to this point


    def process_chunk(
            x_start, x_end,
            y_start, y_end):
        slice_dict = {
            'time': slice(0, len(ds_ldasout.time)),
            'x': slice(x_start, x_end),
            'y': slice(y_start, y_end), }
        _ = (
            da_soil_calc_monthly
            .isel(**slice_dict)
            .to_dataset()
            .to_zarr(file_output, region=slice_dict, consolidated=True))
        return None


    dim_chunk_bounds = {}
    for dd in ['time', 'x', 'y']:
        dim_chunk_bounds[dd] = (
            np.cumsum(
                (0,) +
                dict(zip(ds_ldasout.SOIL_M.dims, ds_ldasout.SOIL_M.chunks))[dd])
            .tolist())

    dict_ii = {}
    for xx in range(len(dim_chunk_bounds['x'])-1):
        if (xx % n_teams) != team_id:
            continue
        for yy in range(len(dim_chunk_bounds['y'])-1):
            dict_ii['x_start'] = dim_chunk_bounds["x"][xx]
            dict_ii['x_end'] = dim_chunk_bounds["x"][xx + 1]
            key_xx = f'x={dict_ii["x_start"]}:{dict_ii["x_end"]}'
            dict_ii['y_start'] = dim_chunk_bounds["y"][yy]
            dict_ii['y_end'] = dim_chunk_bounds["y"][yy + 1]
            key_yy = f'y={dict_ii["y_start"]}:{dict_ii["y_end"]}'
            key_ii = f'{key_xx},{key_yy}'
            log_list = read_log_to_list(file_comp_log)
            if key_ii in log_list:
                continue
            print('\n', flush=True)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
            print(client, flush=True)
            print(key_ii, flush=True)
            timer_start = time.perf_counter()
            process_chunk(**dict_ii)
            append_key_to_log(key_ii, file_comp_log)
            timer_end = time.perf_counter()
            print(f"Chunk took: {timer_end - timer_start:0.4f} seconds", flush=True)

    sys.exit(0)        
