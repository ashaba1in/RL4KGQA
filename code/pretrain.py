import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm.notebook import tqdm 

from drive.MyDrive.Diploma.load_config import config

def pad_batch(batch, e_pad_idx, r_pad_idx):
    batch_size = len(batch)
    entities, relations, labels = zip(*batch)

    padded_entities = pad_sequence(entities, batch_first=True, padding_value=e_pad_idx)
    padded_relations = pad_sequence(relations, batch_first=True, padding_value=r_pad_idx)
    
    return padded_entities, padded_relations, labels


def train_iteration(model, batch, criterion, env):
    device = next(model.parameters()).device
    batch_size = len(batch)
    
    entities, _, _ = zip(*batch)
    cur_entities = [e[-1].item() for e in entities]
    possible_actions = env.get_possible_actions(cur_entities, ban=False)

    trace_entities, trace_relations, actions = pad_batch(batch, model.entity_pad_idx, model.relation_pad_idx)
    trace_entities = trace_entities.to(device)
    trace_relations = trace_relations.to(device)

    labels = []
    for i in range(batch_size):
        if config['only_relations']:
            labels.append(possible_actions[i].index(actions[i]))
        else:
            pos_actions = list(zip(*possible_actions[i]))
            labels.append(pos_actions.index(actions[i]))
    labels = torch.tensor(labels, device=device)

    action_keys = []
    idx = 0
    for actions in possible_actions:
        if config['only_relations']:
            rels = actions
        else:
            rels = actions[0]
        action_keys.append(np.arange(idx, idx + len(rels)))
        idx += len(rels)

    if config['only_relations']:
        flat_possible_actions = torch.LongTensor([a for actions in possible_actions for a in actions]).to(device)
    else:
        flat_possible_actions = (
            torch.LongTensor([r for actions in possible_actions for r in actions[0]]).to(device),
            torch.LongTensor([e for actions in possible_actions for e in actions[1]]).to(device)
        )

    output = model(trace_entities, trace_relations, flat_possible_actions, action_keys)
    loss = 0
    correct = []
    for i in range(batch_size):
        loss += criterion(output[i].unsqueeze(0), labels[i].unsqueeze(0))
        correct.append((output[i].argmax() == labels[i]).detach().cpu().float())

    loss /= batch_size

    return loss, correct


def pretrain(model, dataset, optimizer, criterion, env):
    device = next(model.parameters()).device
    batch_size = env.batch_size

    model.train()

    epoch_loss = []
    accuracy = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        loss, correct = train_iteration(model, batch, criterion, env)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        epoch_loss.append(loss.item())
        accuracy.extend(correct)

    return np.mean(epoch_loss), np.mean(accuracy)


@torch.no_grad()
def evaluate(model, dataset, criterion, env):
    device = next(model.parameters()).device
    batch_size = env.batch_size

    model.eval()

    epoch_loss = []
    accuracy = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        loss, correct = train_iteration(model, batch, criterion, env)
        
        epoch_loss.append(loss.item())
        accuracy.extend(correct)

    return np.mean(epoch_loss), np.mean(accuracy)
