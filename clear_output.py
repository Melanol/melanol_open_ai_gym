import glob
import os
import shutil


# Delete saved models
for file in glob.glob("*.zip"):
    os.remove(file)

# Delete tensorboard logs
try:
    shutil.rmtree('./tensorboard_log')
except FileNotFoundError:
    pass
