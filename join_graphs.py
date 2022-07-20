"""Combines tensorboard graphs into 1 so that it looks like training never stopped."""

import os
import shutil

ALGO_FOLDER = 'LunarLander-v2'
TENSORBOARD_LOG = 'tensorboard_log'

os.chdir(f'./{ALGO_FOLDER}/{TENSORBOARD_LOG}')
all_runs = os.listdir()
algo_names = set()
for run in all_runs:
    if '_' in run:
        algo_names.add(run[:run.index('_')])
    else:
        algo_names.add(run)

# Make folders
for algo in algo_names:
    try:
        os.mkdir(f'./{algo}')
    except FileExistsError:
        pass

# Move logs
for run in all_runs:
    if '_' in run:
        file_path = f'./{run}/' + os.listdir(f'./{run}')[0]
        target_dir = f'./{run[:run.index("_")]}'
        shutil.move(file_path, target_dir)
        os.rmdir(f'./{run}')
