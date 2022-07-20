"""Combines tensorboard graphs into 1 so that it looks like training never stopped."""

import os
import shutil

ALGO_FOLDER = 'LunarLander-v2'
TENSORBOARD_LOG = 'tensorboard_log'

os.chdir(f'./{ALGO_FOLDER}/{TENSORBOARD_LOG}')
all_runs = os.listdir()
algo_names = {el[:el.index('_')] for el in all_runs}

# Make folders
for algo in algo_names:
    os.mkdir(f'./{algo}')

# Move logs
for run in all_runs:
    file_path = f'./{run}/' + os.listdir(f'./{run}')[0]
    target_dir = f'./{run[:run.index("_")]}'
    shutil.move(file_path, target_dir)
    os.rmdir(f'./{run}')
