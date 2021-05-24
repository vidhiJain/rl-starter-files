import gym
import gym_minigrid


def make_env(env_key, seed=None, top_down=False):
    env = gym.make(env_key)
    if top_down:
        env = gym_minigrid.wrappers.FullyObsWrapper(env)
    env.seed(seed)
    return env
