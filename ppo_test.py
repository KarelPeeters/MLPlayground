import itertools

import gym


def main():
    actor_count = 16

    all_envs = []
    all_obs = []

    for _ in range(actor_count):
        env = gym.make("HalfCheetah-v4")
        all_envs.append(env)
        all_obs.append(env.reset())

    for i in itertools.count():
        for env, obs in zip(all_envs, all_obs):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    main()
