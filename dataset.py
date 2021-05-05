from torchtext import data

import numpy as np
import torch

import random


class KGDataset():
    def __init__(self, dataset, env, shuffle=True):

        def make_sample(entities, relations, label):
            sample_e = entities
            sample_r = list(relations) + [len(env.relations)]  # <cls> token
            return torch.tensor(sample_e), torch.tensor(sample_r), label

        self.examples = []
        for _data in dataset:
            entity, relation = zip(*_data)
            for i in range(2, len(entity) + 1):
                self.examples.append(
                    make_sample(entity[:i], relation[:i-1], relation[i-1])
                )
        if shuffle:
            random.shuffle(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


# class CustomIterator(data.Iterator):
#     def __init__(self, dataset, *args, **kwargs):
#         super().__init__(dataset, *args, **kwargs)

#         self.max_in_batch = 0

#         self.sort_key = lambda x: len(x[0])

#     def _batch_size_fn(self, new, count, *args):
#         if count == 1:
#             self.max_in_batch = 0

#         self.max_in_batch = max(self.max_in_batch, len(new[0]))
#         n_elements = count * self.max_in_batch

#         return n_elements

#     def pool(self, d, random_shuffler):
#         for p in data.batch(d, self.batch_size * 100):
#             p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self._batch_size_fn)
#             for b in random_shuffler(list(p_batch)):
#                 yield b

#     def create_batches(self):
#         if self.train:
#             self.batches = self.pool(self.data(), self.random_shuffler)
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size, self._batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))

#     def __len__(self):
#         return len(self.dataset)
