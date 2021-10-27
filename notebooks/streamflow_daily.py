# coding: utf-8

# Streamflow daily averages for the retrospective
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

n_teams = 5
n_workers = 12
dir_proj = pathlib.Path('/glade/scratch/jamesmcc/streamflow_daily/')
file_comp_log = dir_proj / 'streamflow_daily_complete.log'
file_zarr = dir_proj / 'streamflow_daily.zarr'
file_chrtout = '/glade/p/datashare/ishitas/nwm_retro_v2.1/chrtout.zarr'


def read_log_to_list(file_log):
    with file_log.open("r") as opened_file:
        line_list = [line.strip() for line in opened_file]
    return line_list


def append_key_to_log(key, file_log):
    with file_log.open('a') as opened_file:
        opened_file.write(f'{key}\n')
    return None


if __name__ == "__main__":
    team_id = int(sys.argv[-1])
    print(f'\nteam_id: {team_id}')

    if (team_id == 0) and (not file_comp_log.exists()):
        file_comp_log.touch()

    # dask probably dosent depend on clock time, but just in case stagger
    time.sleep(4 * team_id)  

    cluster = PBSCluster(
        cores=1,
        queue='casper', 
        project='NRAL0017', 
        memory='20GB', 
        walltime='04:00:00',
        death_timeout=75)

    cluster.scale(jobs=n_workers)
    client = Client(cluster)

    # Secure some minimum workers before proceeding
    n_workers_start =math.floor(n_workers / 2)
    n_workers_active = 0
    while n_workers_active < n_workers_start:
        client_repr = repr(client)
        n_workers_active = int(client_repr.split('processes=')[1].split(' ')[0])
        print(f'Have {n_workers_active}/{n_workers_start} '
              f'dask workers needed to start job.')
        time.sleep(5)

    dash_link = client.dashboard_link
    print(dash_link)
    hostname = socket.gethostname()
    user = os.environ["USER"]
    port = dash_link.split(':')[-1].split('/')[0]
    print(f"ssh -NL {port}:{hostname}:{port} {user}@casper.ucar.edu", flush=True)
    print(client, flush=True)

    da_streamflow = (
        xr.open_zarr(file_chrtout)
        .streamflow.drop(['longitude', 'latitude', 'elevation', 'gage_id', 'order']))

    da_streamflow_daily = (
        da_streamflow
        .resample(time="1D").mean()
        .chunk({'time': 15310, 'feature_id': 10000}))

    if not file_zarr.exists():
        if team_id == 0:
            da_streamflow_daily.to_dataset().to_zarr(
                file_zarr, compute=False, consolidated=True)
            # This took 4min for the full dataset
        else:
            raise FileExistsError(f'No such file: {file_zarr}')
            # Could do a "lock" file here or just run a single scipt
            # once off-line to this point


    def process_feat_chunk(feat_start, feat_end):
        slice_dict = {
            'time': slice(0, 15310),
            'feature_id': slice(feat_start, feat_end)}
        _ = (
            da_streamflow_daily
            .isel(**slice_dict)
            .to_dataset()
            .to_zarr(file_zarr, region=slice_dict, consolidated=True))
        return None


    feat_chunk_bounds = (
        np.cumsum(
            (0,) +
            dict(zip(da_streamflow.dims, da_streamflow.chunks))['feature_id'])
        .tolist())

    for ii in range(len(feat_chunk_bounds)-1):
        if (ii % n_teams) == team_id:
            key_ii = f'{feat_chunk_bounds[ii]}:{feat_chunk_bounds[ii+1]}'
            log_list = read_log_to_list(file_comp_log)
            if key_ii in log_list:
                continue
            print('\n')
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(key_ii, flush=True)
            timer_start = time.perf_counter()
            process_feat_chunk(feat_chunk_bounds[ii], feat_chunk_bounds[ii + 1])            
            append_key_to_log(key_ii, file_comp_log)
            timer_end = time.perf_counter()
            print(f"Chunk took: {timer_end - timer_start:0.4f} seconds")            

    sys.exit(0)        
