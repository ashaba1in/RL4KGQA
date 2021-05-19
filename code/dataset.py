from torchtext import data

import numpy as np
import torch

import random

from drive.MyDrive.Diploma.load_config import config


class KGDataset():
    def __init__(self, dataset, env, shuffle=True):

        def make_sample(entities, relations, next_r, next_e=None):
            sample_e = entities
            sample_r = list(relations) + [env.CLS]
            label = next_r if next_e is None else (next_r, next_e)
            return torch.tensor(sample_e), torch.tensor(sample_r), label

        self.examples = []
        for _data in dataset:
            entity, relation = zip(*_data)
            for i in range(2, len(entity) + 1):
                if config['only_relations']:
                    self.examples.append(
                        make_sample(entity[:i], relation[:i-1], relation[i-1])
                    )
                else:
                    if i == len(entity):
                        next_r = env.STOP_IDX
                        next_e = entity[i-1]
                    else:
                        next_r = relation[i-1]
                        next_e = entity[i]
                    self.examples.append(
                        make_sample(entity[:i], relation[:i-1], next_r, next_e)
                    )
        if shuffle:
            random.shuffle(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)