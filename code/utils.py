import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from drive.MyDrive.Diploma.model import Encoder, Transformer
from drive.MyDrive.Diploma.load_config import config


def read_dataset_from_file(path, stop_idx):
    dataset = []
    with open(path, 'r') as f:
        path_trace = []
        for line in f:
            e, r = [int(x) for x in line.strip().split()]
            path_trace.append((e, r))
            if r == stop_idx:
                dataset.append(path_trace)
                path_trace = []

    return dataset


def add_to_file(path, path_trace):
    with open(path, 'a') as f:
        for e_r in path_trace:
            f.write('{} {}\n'.format(*e_r))


def remove_cycles(path_trace):
    res_entities = []
    res_relation = []
    ent2idx = {}
    for i, (e, r) in enumerate(path_trace[1:-1]):
        if e in res_entities:
            res_entities = res_entities[:ent2idx[e] + 1]
            res_relation = res_relation[:ent2idx[e]]
            res_relation.append(r)
        else:
            ent2idx[e] = len(res_entities)
            res_entities.append(e)
            res_relation.append(r)

    return [path_trace[0]] + list(zip(res_entities, res_relation)) + [path_trace[-1]]


def walk(s, env):
    path_trace = [(s[0][0], s[1][0])]
    for i in range(256):
        possible_actions = env.get_possible_actions([s[0][0]])[0]
        if isinstance(possible_actions[0], tuple):
            pos_actions = list(zip(*possible_actions))
            a = pos_actions[np.random.randint(len(pos_actions))]
            rel = a[0]
        else:
            a = np.random.choice(possible_actions)
            rel = a

        path_trace.append((s[0][0], rel))

        s, rewards, _, _ = env.step([a])

        if rewards[0] == 1:
            return remove_cycles(path_trace)

        if rel == env.STOP_IDX:
            return None

        if s[0] == env.target_entities:
            path_trace.append((s[0][0], env.STOP_IDX))
            return remove_cycles(path_trace)

    return None


def create_random_dataset(env, out_file, size=100):
    env.batch_size = 1

    dataset = []
    while len(dataset) < size:
        s = env.reset()

        path_trace = walk(s, env)
        if path_trace is not None:
            dataset.append(path_trace)
            add_to_file(out_file, path_trace)

    return dataset


def create_test_dataset(env, test_env, out_file):
    env.batch_size = 1

    dataset = []
    for e1, r, e2 in test_env.triplets:
        env.e_ss = [e1]
        env.qs = [r]
        env.e_ts = [e2]
        env.es = [e1]

        s = ([e1], [r])

        path_trace = walk(s, env)
        if path_trace is not None:
            dataset.append(path_trace)
            add_to_file(out_file, path_trace)

    return dataset


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def create_model(entity_input_dim, relation_input_dim, output_dim, entity_pad_idx, relation_pad_idx,
                 hid_dim=128, n_enc_layers=3, n_enc_heads=8, enc_pf_dim=256, enc_dp=0.1,
                 device=torch.device('cuda')):

    enc = Encoder(entity_input_dim, relation_input_dim, hid_dim, n_enc_layers, n_enc_heads, enc_pf_dim, enc_dp)

    model = Transformer(enc, entity_input_dim, relation_input_dim,
                        hid_dim, output_dim, entity_pad_idx, relation_pad_idx).to(device)
    model.apply(initialize_weights)

    return model.to(device)


def linear_combination(x, y, epsilon): 
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
