"""Run "tensorboard --logdir ./tensorboard_log"."""

# I have a feeling that tensorboard does not work if there is not at least 10000 steps

import gym

from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

FILENAME = 'A2C_MlpPolicy_CartPole-v1'
LOAD = 0
SAVE = 0
TENSORBOARD_LOG = "./tensorboard_log/"

env = gym.make('CartPole-v1')

if LOAD:
    model = A2C.load(FILENAME, env, verbose=1, tensorboard_log=TENSORBOARD_LOG,
                     policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
else:
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=TENSORBOARD_LOG,
                policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))

model.learn(total_timesteps=1000)

if SAVE:
    model.save(FILENAME)

# Render
obs = env.reset()
for i in range(0):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
