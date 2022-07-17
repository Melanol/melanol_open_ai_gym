import multiprocessing as mp
from os.path import exists
import subprocess
import time
import webbrowser

import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

POLICY = 'MlpPolicy'
STEPS = 200000
LOAD = True
SAVE = True
VIDEO_STEPS = 1000
SAVE_VIDEO = False
VERBOSE = False  # Outputs progress into console
ENVNAME = 'CartPole-v1'
LAUNCH_TENSORBOARD = True  # If true, will have to terminate the process manually
TENSORBOARD_LOG = f"./{ENVNAME}/tensorboard_log/"


def model_process():
    # Create env and model
    env = gym.make(ENVNAME)
    if LOAD and exists(f'./{ENVNAME}/{ENVNAME}.zip'):
        model = A2C.load(f'./{ENVNAME}/{ENVNAME}', env, verbose=VERBOSE, tensorboard_log=TENSORBOARD_LOG,
                         policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
        print('Model loaded')
    else:
        model = A2C(POLICY, env, verbose=VERBOSE, tensorboard_log=TENSORBOARD_LOG,
                    policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
        print('New model created')

    # Learn and save
    model.learn(total_timesteps=STEPS)
    print('Learning complete')
    if SAVE:
        model.save(ENVNAME)
        print('Saving complete')

    # Video
    images = []
    obs = env.reset()
    for i in range(VIDEO_STEPS):
        images.append(env.render(mode='rgb_array'))
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    if SAVE_VIDEO:
        print('Making video')
        dpi = 70
        height, width, _ = images[0].shape
        frames = []
        fig = plt.figure(figsize=(width/dpi, height/dpi), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        for i in range(len(images)):
            frames.append([plt.imshow(images[i], animated=True)])
        ani = animation.ArtistAnimation(fig, frames, interval=1000/30, blit=True)
        ani.save(f'{ENVNAME}.mp4')
        print('Video saved')

    print('Done')


def start_tensorboard():
    cmd = f'tensorboard --logdir {TENSORBOARD_LOG}'
    subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
    print('Tensorboard launched')


def open_tensorboard():
    time.sleep(3)  # Wait until tensorboard is online
    url = "http://localhost:6006/ "
    webbrowser.open(url, new=0, autoraise=True)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.Process(target=model_process).start()
    if LAUNCH_TENSORBOARD:
        mp.Process(target=start_tensorboard).start()
        mp.Process(target=open_tensorboard).start()
