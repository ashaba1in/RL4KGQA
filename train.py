import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import time

from drive.MyDrive.Diploma.dataset import KGDataset
from drive.MyDrive.Diploma.load_config import config


EPS = float(np.finfo(float).eps)


def train_emb(model, dataset, optimizer, criterion, batch_size):
    device = next(model.parameters()).device

    model.train()

    epoch_loss = []
    accuracy = []
    all_ranks = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        entities, relations, labels = zip(*batch)

        entities = torch.tensor(entities, device=device)
        relations = torch.tensor(relations, device=device)
        labels = torch.tensor(labels, device=device)

        optimizer.zero_grad()

        output = model(entities, relations)

        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        ranks = (output >= output.gather(1, labels.view(-1, 1))).sum(1)
        all_ranks.extend(ranks.detach().cpu())
        accuracy.append((output.argmax(axis=1) == labels).detach().cpu().float().mean())
        epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(accuracy), all_ranks


@torch.no_grad()
def evaluate_emb(model, dataset, criterion, batch_size):
    device = next(model.parameters()).device

    model.eval()

    epoch_loss = []
    accuracy = []
    all_ranks = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        entities, relations, labels = zip(*batch)

        entities = torch.tensor(entities, device=device)
        relations = torch.tensor(relations, device=device)
        labels = torch.tensor(labels, device=device)

        output = model(entities, relations)

        loss = criterion(output, labels)

        ranks = (output >= output.gather(1, labels.view(-1, 1))).sum(1)
        all_ranks.extend(ranks.detach().cpu())
        accuracy.extend((output.argmax(axis=1) == labels).detach().cpu().float())
        epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(accuracy), all_ranks


def pad_batch(batch, e_pad_idx, r_pad_idx):
    batch_size = len(batch)
    entities, relations, labels = zip(*batch)

    padded_entities = pad_sequence(entities, batch_first=True, padding_value=e_pad_idx)
    padded_relations = pad_sequence(relations, batch_first=True, padding_value=r_pad_idx)
    
    return padded_entities, padded_relations, torch.tensor(labels)


def pretrain(model, dataset, optimizer, criterion, batch_size):
    device = next(model.parameters()).device

    model.train()

    epoch_loss = []
    accuracy = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        entities, relations, labels = pad_batch(batch, model.entity_pad_idx, model.relation_pad_idx)

        entities = entities.to(device)
        relations = relations.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(entities, relations)

        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        accuracy.append((output.argmax(axis=1) == labels).detach().cpu().float().mean())
        epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(accuracy)


@torch.no_grad()
def evaluate(model, dataset, criterion, batch_size):
    device = next(model.parameters()).device

    model.eval()

    epoch_loss = []
    accuracy = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        entities, relations, labels = pad_batch(batch, model.entity_pad_idx, model.relation_pad_idx)

        entities = entities.to(device)
        relations = relations.to(device)
        labels = labels.to(device)

        output = model(entities, relations)

        loss = criterion(output, labels)

        accuracy.append((output.argmax(axis=1) == labels).detach().cpu().float().mean())
        epoch_loss.append(loss.item())

    return np.mean(epoch_loss), np.mean(accuracy)


def len_penalty(step):
    return np.arctan(step / 10) / 2


def run_episode(env, agent):
    device = next(agent.parameters()).device
    
    cur_entity, relation = env.reset()

    path_trace_e = np.vstack((
        cur_entity,
        cur_entity
    )).T
    path_trace_r = np.array([relation]).T

    step = 0
    num_correct = 0
    dones = np.zeros(env.batch_size, dtype=bool)  # all False at the start

    saved_probs = [[] for _ in range(env.batch_size)]
    saved_rewards = [[] for _ in range(env.batch_size)]
    saved_actions = [[] for _ in range(env.batch_size)]
    saved_values = [[] for _ in range(env.batch_size)]

    while step < config['max_rollout']:
        entities = torch.tensor(path_trace_e, device=device)
        relations = torch.tensor(
            np.hstack((
                path_trace_r,
                np.expand_dims([env.CLS] * len(path_trace_r), 1)
            )),
            device=device
        )

        # select action
        preds = agent(entities, relations)
        if config['rl_method'] == 'A2C':
            preds, values = preds
            values = values.detach().cpu().numpy().squeeze(1)
            for i, j in enumerate(np.where(~dones)[0]):
                saved_values[j].append(values[i])

        possible_actions = env.get_possible_actions(path_trace_e[:, -1])

        next_relations = []
        for i, j in zip(range(len(preds)), np.where(~dones)[0]):
            # TODO compare entropy over only possible actions
            # all_probs = F.softmax(preds[i], dim=0)
            # saved_probs[j].append(all_probs)  # save probs

            probs = F.softmax(preds[i][possible_actions[i]], dim=0)

            idx = Categorical(probs).sample()
            action = possible_actions[i][idx.item()]
            next_relations.append(action)
            saved_actions[j].append(idx)  # we save only idx in possible actions
            saved_probs[j].append(probs)  # save probs

        actions = np.full(env.batch_size, env.STOP_IDX)  # fill with STOP
        actions[~dones] = next_relations

        # make a step
        obs, rewards, upd_dones, info = env.step(actions)

        # save rewards
        for i in np.where(~dones)[0]:
            num_correct += float(rewards[i] == 1)
            if config['len_penalty'] and rewards[i] > 0:
                rewards[i] -= len_penalty(step)
            saved_rewards[i].append(rewards[i])

        new_dones = upd_dones ^ dones
        not_done = ~new_dones[~dones]
        dones = upd_dones

        # expand trajectory
        entities, _ = obs

        if config['rl_method'] == 'A2C' and np.any(~not_done):
            entities_ = np.hstack((
                path_trace_e[~not_done],
                np.expand_dims(entities[new_dones], 1)
            ))
            relations_ = np.hstack((
                path_trace_r[~not_done],
                np.expand_dims([env.STOP_IDX] * sum(~not_done), 1),
                np.expand_dims([env.CLS] * sum(~not_done), 1)
            ))
            entities_ = torch.tensor(entities_, device=device)
            relations_ = torch.tensor(relations_, device=device)

            _, values = agent(entities_, relations_)
            values = values.detach().cpu().numpy().squeeze(1)
            for i, j in enumerate(np.where(new_dones)[0]):
                saved_values[j].append(values[i])

        path_trace_e = np.hstack((
            path_trace_e[not_done],
            np.expand_dims(entities[~dones], 1)
        ))
        path_trace_r = np.hstack((
            path_trace_r[not_done],
            np.expand_dims(actions[~dones], 1)
        ))

        if np.all(dones):
            break

        step += 1

    # if config['rl_method'] == 'A2C' and np.any(~dones):
    #     entities_ = np.hstack((
    #         path_trace_e,
    #         np.expand_dims(entities[~dones], 1)
    #     ))
    #     relations_ = np.hstack((
    #         path_trace_r,
    #         np.expand_dims([env.STOP_IDX] * sum(~dones), 1),
    #         np.expand_dims([env.CLS] * sum(~dones), 1)
    #     ))
    #     entities_ = torch.tensor(entities_, device=device)
    #     relations_ = torch.tensor(relations_, device=device)

    #     _, values = agent(entities_, relations_)
    #     values = values.detach().cpu().numpy().squeeze(1)
    #     for i, j in enumerate(np.where(~dones)[0]):
    #         saved_values[j].append(values[i])

    if config['normalize_reward']:
        for i in range(len(saved_rewards)):
            rewards = np.array(saved_rewards[i])
            saved_rewards[i] = (rewards - rewards.mean()) # / (rewards.std() + EPS)

    assert [len(probs) for probs in saved_probs] == \
           [len(rewards) for rewards in saved_rewards] ==\
           [len(actions) for actions in saved_actions]

    episod_info = {
        'probs': saved_probs,
        'rewards': saved_rewards,
        'actions': saved_actions,
        'num_steps': step,
        'num_correct': num_correct,
        'values': saved_values
    }
    return episod_info


def calc_G(rewards):
    gs = [rewards[-1]]
    for i in range(len(rewards) - 2, -1, -1):
        gs.append(rewards[i] + config['gamma'] * gs[-1])

    return reversed(gs)

def calc_advantage(rewards, values):
    Qvals = np.zeros_like(rewards)
    val = values[-1]
    for i in reversed(range(len(rewards))):
        val = rewards[i] + config['gamma'] * val
        Qvals[i] = val

    advantage = Qvals - np.array(values[:-1])
    return advantage

def train(agent, optimizer, episod_info):
    """
    saved_probs: probs for all actions for each step (need for entropy regularization)
    """
    device = next(agent.parameters()).device

    saved_probs = episod_info['probs']
    saved_rewards = episod_info['rewards']
    saved_actions = episod_info['actions']
    saved_values = episod_info['values']

    if config['rl_method'] == 'REINFORCE':
        vals = [calc_G(rewards) for rewards in saved_rewards]
    elif config['rl_method'] == 'A2C':
        vals = [calc_advantage(rewards, values) for rewards, values in zip(saved_rewards, saved_values)]
    
    flat_probs = [prob for probs in saved_probs for prob in probs]
    flat_vals = [val for vals_ in vals for val in vals_]
    flat_actions = [a for actions in saved_actions for a in actions]

    assert len(flat_probs) == len(flat_vals) == len(flat_actions)

    entropy = []
    policy_loss = []
    critic_loss = []
    for probs, a, val in zip(flat_probs, flat_actions, flat_vals):
        policy_loss.append(-torch.log(probs[a]) * val)
        entropy.append((torch.log(probs) * probs).sum())
        critic_loss.append(val ** 2)

    loss = sum(policy_loss) / len(policy_loss)
    loss += config['entropy_coef'] * sum(entropy) / len(entropy)
    if config['rl_method'] == 'A2C':
        loss += 0.5 * sum(critic_loss) / len(critic_loss)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 1)
    optimizer.step()

    return loss.item()
