import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import time

from drive.MyDrive.Diploma.load_config import config


EPS = float(np.finfo(float).eps)


def len_penalty(step):
    return np.arctan(step / 10) / 6


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
    
    len_pens = []
    steps = []

    while step < config['max_rollout']:
        # get action distribution
        trace_entities = torch.tensor(path_trace_e, device=device)
        trace_relations = torch.tensor(
            np.hstack((
                path_trace_r,
                np.expand_dims([env.CLS] * len(path_trace_r), 1)
            )),
            device=device
        )

        # [trace_num, pos_act_i]
        possible_actions = env.get_possible_actions(path_trace_e[:, -1], np.where(~dones)[0], ban=True)

        # [trace_num, pos_act_i]
        action_keys = []
        idx = 0
        for actions in possible_actions:
            if config['only_relations']:
                rels = actions
            else:
                rels = actions[0]
            action_keys.append(np.arange(idx, idx + len(rels)))
            idx += len(rels)

        # [sum(pos_act_i)]
        if config['only_relations']:
            flat_possible_actions = torch.LongTensor([a for actions in possible_actions for a in actions]).to(device)
        else:
            flat_possible_actions = (
                torch.LongTensor([r for actions in possible_actions for r in actions[0]]).to(device),
                torch.LongTensor([e for actions in possible_actions for e in actions[1]]).to(device)
            )

        preds = agent(trace_entities, trace_relations, flat_possible_actions, action_keys)

        if config['rl_method'] == 'A2C':
            preds, values = preds
            values = values.detach().cpu().numpy().squeeze(1)
            for i, j in enumerate(np.where(~dones)[0]):
                saved_values[j].append(values[i])

        # select action
        next_actions = []
        for i, j in zip(range(len(preds)), np.where(~dones)[0]):
            action_dist = F.softmax(preds[i], dim=0)

            idx = Categorical(action_dist).sample()

            if config['only_relations']:
                action = possible_actions[i][idx.item()]
            else:
                rs, es = possible_actions[i]
                action = (rs[idx.item()], es[idx.item()])

            next_actions.append(action)
            saved_actions[j].append(idx)  # we save only idx in possible actions
            saved_probs[j].append(action_dist)  # save probs

        # make a step
        obs, rewards, upd_dones, _ = env.step(next_actions, np.where(~dones)[0])

        # save rewards
        for i in np.where(~dones)[0]:
            if (step == config['max_rollout'] - 1 and config['static_relation'] == 'repeat') or\
                config['static_relation'] == 'stop':
                num_correct += float(rewards[i] == 1)
            if config['len_penalty'] and rewards[i] > 0:
                rewards[i] -= len_penalty(step)
                len_pens.append(len_penalty(step))
            saved_rewards[i].append(rewards[i])

        new_dones = upd_dones ^ dones
        not_done = ~new_dones[~dones]
        dones = upd_dones

        steps.extend([step] * sum(new_dones))

        # expand trajectory
        entities, _ = obs

        # if config['rl_method'] == 'A2C' and np.any(~not_done):
        #     entities_ = np.hstack((
        #         path_trace_e[~not_done],
        #         np.expand_dims(entities[new_dones], 1)
        #     ))
        #     relations_ = np.hstack((
        #         path_trace_r[~not_done],
        #         np.expand_dims([env.STOP_IDX] * sum(~not_done), 1),
        #         np.expand_dims([env.CLS] * sum(~not_done), 1)
        #     ))
        #     entities_ = torch.tensor(entities_, device=device)
        #     relations_ = torch.tensor(relations_, device=device)

        #     _, values = agent(entities_, relations_)
        #     values = values.detach().cpu().numpy().squeeze(1)
        #     for i, j in enumerate(np.where(new_dones)[0]):
        #         saved_values[j].append(values[i])


        path_trace_e = np.hstack((
            path_trace_e[not_done],
            np.expand_dims(entities[~dones], 1)
        ))

        if not config['only_relations']:
            next_actions, _ = zip(*next_actions)
        next_actions = np.array(next_actions)
        path_trace_r = np.hstack((
            path_trace_r[not_done],
            np.expand_dims(next_actions[not_done], 1)
        ))

        step += 1

        if np.all(dones):
            break

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

    steps.extend([step] * sum(~dones))

    assert [len(probs) for probs in saved_probs] == \
           [len(rewards) for rewards in saved_rewards] ==\
           [len(actions) for actions in saved_actions]

    episod_info = {
        'probs': saved_probs,
        'rewards': saved_rewards,
        'actions': saved_actions,
        'num_steps': step,
        'num_correct': num_correct,
        'values': saved_values,
        'mean_steps': np.mean(steps),
        'len_pen': np.mean(len_pens)
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

    if config['normalize_reward']:
        for i in range(len(vals)):
            rewards = np.array(vals[i])
            vals[i] = (rewards - rewards.mean()) / (rewards.std() + EPS)
    
    flat_probs = [act_probs for trace_probs in saved_probs for act_probs in trace_probs]
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
