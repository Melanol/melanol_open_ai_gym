"""Run "tensorboard --logdir ./tensorboard_log"."""

# A2C does not converge properly even with the recommended RMSpropTFLike

from os.path import exists

import gym

from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

POLICY = 'MlpPolicy'
STEPS = 10000
DEMO_STEPS = 0
LOAD = True
SAVE = True
VERBOSE = False  # Outputs progress into console
FILENAME = 'A2C_MlpPolicy_CartPole-v1'
TENSORBOARD_LOG = "./tensorboard_log/"

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

# Demo
obs = env.reset()
for i in range(DEMO_STEPS):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
