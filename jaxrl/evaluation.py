from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(discount, agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': [], 'oracle_q': []}
    successes = None
    for e in range(num_episodes):
        observation, done = env.reset(), False
        stats['oracle_q'].append([])
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, done, info = env.step(action)
            stats['oracle_q'][e].append(reward)
        for k in stats.keys():
            if k != 'oracle_q':
                stats[k].append(info['episode'][k])
        stats['oracle_q'][e] = np.sum([r * discount**k for k, r in enumerate(stats['oracle_q'][e])])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats
