import numpy as np

def mmr(ranks):
    return np.mean(1 / np.array(ranks))

def hit_k(ranks, k=10):
    return np.mean(np.array(ranks) <= k)
