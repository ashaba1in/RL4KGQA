import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from drive.MyDrive.Diploma.model import Encoder, Transformer


def read_from_file(path, env):
    dataset = []
    with open(path, 'r') as f:
        path_trace = []
        for line in f:
            e, r = [int(x) for x in line.strip().split()]
            path_trace.append((e, r))
            if r == env.STOP_IDX:
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
            res_relation = res_relation[:ent2idx[e] + 1]
        else:
            ent2idx[e] = len(res_entities)
            res_entities.append(e)
            res_relation.append(r)

    return [path_trace[0]] + list(zip(res_entities, res_relation)) + [path_trace[-1]]


def create_random_dataset(env, out_file, size=100):
    env.batch_size = 1

    dataset = []
    while len(dataset) < size:
        s = env.reset()

        path_trace = [(s[0][0], s[1][0])]

        for i in range(256):
            a = np.random.choice(env.get_possible_actions(s[0])[0])
            path_trace.append((s[0][0], a))

            next_s, reward, done, _ = env.step([a])

            s = next_s

            if reward[0] == 1:
                path_trace = remove_cycles(path_trace)
                dataset.append(path_trace)
                add_to_file(out_file, path_trace)
                break
            if done[0]:
                break

    return dataset


def create_test_dataset(env, test_env, out_file):
    env.batch_size = 1

    dataset = []
    for e1, r, e2 in test_env.allowed_triplets:
        env.current_entities = [e1]
        env.target_relations = [r]
        env.target_entities = [e2]

        s = ([e1], [r])

        path_trace = [(s[0][0], s[1][0])]

        for i in range(256):
            a = np.random.choice(env.get_possible_actions([s[0][0]])[0])
            path_trace.append((s[0][0], a))

            next_s, reward, done, _ = env.step([a])

            s = next_s

            if reward[0] == 1:
                path_trace = remove_cycles(path_trace)
                dataset.append(path_trace)
                add_to_file(out_file, path_trace)
                break
            if done[0]:
                break

    return dataset


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def create_model(entity_input_dim, relation_input_dim, output_dim, entity_pad_idx, relation_pad_idx,
                 hid_dim=512, n_enc_layers=3, n_enc_heads=8, enc_pf_dim=1024, enc_dp=0.1,
                 device=torch.device('cuda')):

    enc = Encoder(entity_input_dim, relation_input_dim, hid_dim, n_enc_layers, n_enc_heads, enc_pf_dim, enc_dp)

    model = Transformer(enc, hid_dim, output_dim, entity_pad_idx, relation_pad_idx).to(device)
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
