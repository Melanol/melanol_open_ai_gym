"""Terminate manually after "Model done", as tensorboard will not terminate itself."""

# A2C does not converge properly even with the recommended RMSpropTFLike

import multiprocessing as mp
from os.path import exists
import subprocess
import time
import webbrowser

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

POLICY = 'MlpPolicy'
STEPS = 10000
DEMO_STEPS = 100
LOAD = True
SAVE = True
VERBOSE = False  # Outputs progress into console
FILENAME = 'CartPole-v1'
LAUNCH_TENSORBOARD = True
TENSORBOARD_LOG = "./tensorboard_log/"


def model_process():
    # Create env and model
    env = gym.make('CartPole-v1')
    if LOAD and exists(FILENAME+'.zip'):
        model = A2C.load(FILENAME, env, verbose=VERBOSE, tensorboard_log=TENSORBOARD_LOG,
                         policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
        print('Model loaded')
    else:
        model = A2C(POLICY, env, verbose=VERBOSE, tensorboard_log=TENSORBOARD_LOG,
                    policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
        print('New model created')

    # Learn and save
    model.learn(total_timesteps=STEPS)
    if SAVE:
        model.save(FILENAME)
        print('Learning complete')

    # Demo
    obs = env.reset()
    for i in range(DEMO_STEPS):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


def start_tensorboard():
    cmd = 'tensorboard --logdir ./tensorboard_log'
    subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
    print('Tensorboard launched')


def open_tensorboard():
    time.sleep(1)  # Wait until tensorboard is online
    url = "http://localhost:6006/ "
    webbrowser.open(url, new=0, autoraise=True)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.Process(target=model_process).start()
    if LAUNCH_TENSORBOARD:
        mp.Process(target=start_tensorboard).start()
        mp.Process(target=open_tensorboard).start()
