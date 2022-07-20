from pprint import pprint
import random

from matplotlib import pyplot as plt
import gym


def all_envs():
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    return env_ids


def print_all_envs():
    pprint(sorted(all_envs()))


def render_env(env_id=random.choice(all_envs())):
    env = gym.make(env_id)
    print()
    print(env_id)
    env.reset()
    image = env.render(mode='rgb_array')
    plt.imshow(image, interpolation='nearest')
    plt.title(env_id)
    plt.show()


def render_video(env_id):
    """I tried making this function universal for all gym envs, but they are too different."""
    env = gym.make(env_id)
    print()
    print(env_id)
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()


if __name__ == '__main__':
    # print_all_envs()
    # render_env('LunarLander-v2')
    render_video('LunarLander-v2')
