import torch
import torch.nn.functional as F

import numpy as np
from tqdm.notebook import tqdm

from drive.MyDrive.Diploma.load_config import config


def remove_banned_triplet(entities, possible_actions, triplet):
    e1, r, e2 = triplet
    for i in range(len(entities)):
        if entities[i] == e1:
            if config['only_relations']:
                if r in possible_actions[i]:
                    possible_actions[i].remove(r)
            else:
                if r in possible_actions[i][0]:
                    idx = possible_actions[i][0].index(r)
                    if possible_actions[i][1][idx] == e2:
                        possible_actions[i][0] = possible_actions[i][0][:idx] + possible_actions[i][0][idx + 1:]
                        possible_actions[i][1] = possible_actions[i][1][:idx] + possible_actions[i][1][idx + 1:]


def top_k(dist, beam_size):
    flat_dist = torch.tensor([d for dist_ in dist for d in dist_])
    log_probs, idxs = torch.topk(flat_dist, beam_size)

    flat_trace_keys = np.array([d for i, dist_ in enumerate(dist) for d in [i] * len(dist_)])

    idxs = np.array(idxs)
    path_trace_idxs = flat_trace_keys[idxs]

    return log_probs, path_trace_idxs, idxs


def extend_path(path_trace_e, path_trace_r, entities, relations):
    if entities is not None:
        path_trace_e = np.hstack((
            path_trace_e,
            np.expand_dims(entities, 1)
        ))
    if relations is not None:
        path_trace_r = np.hstack((
            path_trace_r,
            np.expand_dims(relations, 1)
        ))
    return path_trace_e, path_trace_r

def bs_step(agent, env, stop_e2_probs, e2_log_probs, path_traces, banned_triplet, lens, rel_paths):
    """
    stop_e2_probs: [entities_num]
    e2_log_probs: [<=beam_size]
    path_traces: (path_trace_e: [<=beam_size], path_trace_r: [<=beam_size])
    """
    device = next(agent.parameters()).device
    beam_size = config['beam_size']

    path_trace_e, path_trace_r = path_traces
    cur_entities = path_trace_e[:, -1]
    
    possible_actions = env.get_possible_actions(cur_entities, ban=False)
    remove_banned_triplet(cur_entities, possible_actions, banned_triplet)

    if len(possible_actions) == 0:
        print(banned_triplet)

    # get action probabilities
    trace_entities = torch.tensor(path_trace_e, device=device)
    trace_relations = torch.tensor(
        np.hstack((
            path_trace_r,
            np.expand_dims([env.CLS] * len(path_trace_r), 1)
        )),
        device=device
    )

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

    with torch.no_grad():
        preds = agent(trace_entities, trace_relations, flat_possible_actions, action_keys)

    # update log action dist
    log_action_dist = []
    possible_actions_num = 0
    
    for i in range(len(preds)):
        action_dist = F.softmax(preds[i].detach().cpu(), dim=0)
        # first action is static
        if stop_e2_probs[cur_entities[i]] < e2_log_probs[i] + np.log(action_dist[0]):
            stop_e2_probs[cur_entities[i]] = e2_log_probs[i] + np.log(action_dist[0])
            lens[cur_entities[i]] = len(path_trace_e[i])
            rel_paths[cur_entities[i]] = path_trace_r[i]

        if config['static_relation'] == 'stop':
            log_action_dist.append(e2_log_probs[i] + torch.log(action_dist[1:]))
            possible_actions_num += len(action_dist) - 1  # minus STOPs
        elif config['static_relation'] == 'repeat':
            log_action_dist.append(e2_log_probs[i] + torch.log(action_dist))
            possible_actions_num += len(action_dist)
        else:
            raise NotImplementedError('Unknown static relation')

    e2_log_probs, path_trace_idxs, idxs = top_k(log_action_dist, min(beam_size, possible_actions_num))
    # path_trace_idxs - paths to expand by relations

    if config['only_relations']:
        if config['static_relation'] == 'stop':
            flat_actions = np.array([a for actions in possible_actions for a in actions[1:]])[idxs]
        elif config['static_relation'] == 'repeat':
            flat_actions = np.array([a for actions in possible_actions for a in actions])[idxs]
    else:
        if config['static_relation'] == 'stop':
            flat_actions = (
                np.array([r for actions in possible_actions for r in actions[0][1:]])[idxs],
                np.array([e for actions in possible_actions for e in actions[1][1:]])[idxs]
            )
        elif config['static_relation'] == 'repeat':
            flat_actions = (
                np.array([r for actions in possible_actions for r in actions[0]])[idxs],
                np.array([e for actions in possible_actions for e in actions[1]])[idxs]
            )

    # update path traces
    path_trace_e = path_trace_e[path_trace_idxs]
    path_trace_r = path_trace_r[path_trace_idxs]

    if config['only_relations']:
        e2_log_probs = e2_log_probs[path_trace_idxs]
        cur_entities = cur_entities[path_trace_idxs]

        possible_actions = env.get_possible_actions(cur_entities, ban=False)
        for i in range(len(possible_actions)):
            assert flat_actions[i] in possible_actions[i]

        log_entity_dist = []
        possible_entities = []
        possible_entities_num = 0

        for i in range(len(flat_actions)):
            entities = env.get_entities(cur_entities[i], flat_actions[i])
            possible_entities_num += len(entities)

            log_prob = e2_log_probs[i] - np.log(len(entities))
            log_entity_dist.append([log_prob] * len(entities))
            possible_entities.append(entities)

        e2_log_probs, path_trace_idxs, idxs = top_k(log_entity_dist, min(beam_size, possible_entities_num))

        flat_entities = np.array([e for entities in possible_entities for e in entities])[idxs]
        
        path_trace_e = path_trace_e[path_trace_idxs]
        path_trace_r = path_trace_r[path_trace_idxs]
        flat_actions = flat_actions[path_trace_idxs]

        path_trace_e, path_trace_r = extend_path(path_trace_e, path_trace_r, flat_entities, flat_actions)
    else:
        path_trace_e, path_trace_r = extend_path(path_trace_e, path_trace_r, flat_actions[1], flat_actions[0])
    
    path_traces = (path_trace_e, path_trace_r)
    return stop_e2_probs, e2_log_probs, path_traces


def beam_search(agent, train_env, test_env, e1, r, e2):
    """
    return rank of e2
    """
    stop_e2_probs = torch.zeros(train_env.entities_num) - np.inf
    e2_log_probs = torch.zeros(1)
    path_trace_e = np.array([[e1, e1]])
    path_trace_r = np.array([[r]])
    path_traces = (path_trace_e, path_trace_r)
    banned_triplet = (e1, r, e2)
    # print(banned_triplet)

    lens = torch.zeros(train_env.entities_num)
    rel_paths = [[] for i in range(train_env.entities_num)]
    for _ in range(config['num_beam_steps']):
        stop_e2_probs, e2_log_probs, path_traces = bs_step(
            agent, train_env, stop_e2_probs, e2_log_probs, path_traces,
            banned_triplet, lens, rel_paths
        )

    for e in range(train_env.entities_num):
        if e != e2:
            if e in test_env.get_entities(e1, r):
                stop_e2_probs[e2] = max(stop_e2_probs[e2], stop_e2_probs[e])
                stop_e2_probs[e] = -np.inf
            elif e in train_env.get_entities(e1, r):
                # stop_e2_probs[e2] = max(stop_e2_probs[e2], stop_e2_probs[e])
                stop_e2_probs[e] = -np.inf
        
    # print(stop_e2_probs.argmax())
    return (stop_e2_probs >= stop_e2_probs[e2]).sum() 


def get_ranks(agent, train_env, test_env):
    # agent.eval()
    triplets = test_env.allowed_triplets

    ranks = []
    for i in tqdm(range(100)):
        idx = np.random.randint(len(triplets))
        e1, r, e2 = triplets[idx]
        if train_env[e1] is None:
            continue

        rank = beam_search(agent, train_env, test_env, e1, r, e2)
        ranks.append(rank)

    return np.array(ranks)

