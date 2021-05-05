import torch
import torch.nn.functional as F

import numpy as np
from tqdm.notebook import tqdm

from drive.MyDrive.Diploma.load_config import config


def top_k(dist, beam_size):
    """
    dist: [<=beam_size, action_num]
    """
    action_num = dist.shape[1] 
    dist = dist.view(-1)
    probs, idxs = torch.topk(dist, beam_size)
    trajectory_idxs = idxs // action_num
    action_idxs = idxs % action_num

    return probs, trajectory_idxs.numpy(), action_idxs.numpy()


def bs_step(agent, env, stop_e2_probs, e2_log_probs, path_traces):
    """
    stop_e2_probs: [entities_num]
    e2_log_probs: [<=beam_size]
    path_traces: (path_trace_e: [<=beam_size], path_trace_r: [<=beam_size])
    """
    device = next(agent.parameters()).device
    beam_size = config['beam_size']

    path_trace_e, path_trace_r = path_traces
    cur_entities = path_trace_e[:, -1]
    
    possible_actions = env.get_possible_actions(cur_entities)

    entities = torch.tensor(path_trace_e, device=device)
    relations = torch.tensor(
        np.hstack((
            path_trace_r,
            np.expand_dims([env.CLS] * len(path_trace_r), 1)
        )),
        device=device
    )
    with torch.no_grad():
        preds = agent(entities, relations).detach().cpu()

    # pick relations
    log_action_prob = torch.zeros_like(preds) - np.inf
    for i in range(len(cur_entities)):
        probs = F.softmax(preds[i][possible_actions[i]], dim=0)
        # first action is STOP
        stop_e2_probs[cur_entities[i]] = max(stop_e2_probs[cur_entities[i]], e2_log_probs[i] + np.log(probs[0]))
        log_action_prob[i][possible_actions[i][1:]] = torch.log(probs[1:])

    # [beam_size, action_num]
    log_action_dist = e2_log_probs.unsqueeze(-1) + log_action_prob

    possible_actions_num = sum([len(actions) - 1 for actions in possible_actions])  # -1 because of STOP action
    r_log_probs, path_trace_idxs, actions = top_k(log_action_dist, min(beam_size, possible_actions_num))
    # path_trace_idxs - paths to expand by relations

    # pick entities
    log_entity_prob = torch.zeros(len(r_log_probs), env.entities_num) - np.inf
    possible_entity_num = 0
    for i in range(len(r_log_probs)):
        e = cur_entities[path_trace_idxs[i]]
        r = env.relations[actions[i]]
        possible_entities = env.get_entities(e, r)
        log_entity_prob[i][possible_entities] = -np.log(len(possible_entities))

        possible_entity_num += len(possible_entities)
    
    # [beam_size, entity_num]
    log_entity_dist = r_log_probs.unsqueeze(-1) + log_entity_prob
    
    e2_log_probs, relation_idxs, entity_idxs = top_k(log_entity_dist, min(beam_size, possible_entity_num))
        
    # we pick idxs from path_trace_idxs to expend paths
    path_trace_idxs = path_trace_idxs[relation_idxs]
    actions = actions[relation_idxs]

    # update path traces
    path_trace_e = path_trace_e[path_trace_idxs]
    path_trace_r = path_trace_r[path_trace_idxs]

    entities = env.entities[entity_idxs]
    path_trace_e = np.hstack((
        path_trace_e,
        np.expand_dims(entities, 1)
    ))
    path_trace_r = np.hstack((
        path_trace_r,
        np.expand_dims(actions, 1)
    ))

    path_traces = (path_trace_e, path_trace_r)

    return stop_e2_probs, e2_log_probs, path_traces


def beam_search(agent, env, e1, r, e2):
    """
    return rank of e2
    """
    stop_e2_probs = torch.zeros(env.entities_num) - np.inf
    e2_log_probs = torch.zeros(1)
    path_trace_e = np.array([[e1, e1]])
    path_trace_r = np.array([[r]])
    path_traces = (path_trace_e, path_trace_r)

    for _ in range(config['num_beam_steps']):
        stop_e2_probs, e2_log_probs, path_traces = bs_step(
            agent, env, stop_e2_probs, e2_log_probs, path_traces
        )

    return (stop_e2_probs >= stop_e2_probs[e2]).sum()


def get_ranks(agent, triplets, env):
    agent.eval()
    ranks = []
    for i in tqdm(range(config['num_rollouts'])):
        idx = np.random.randint(len(triplets))
        e1, r, e2 = triplets[idx]
        if env[e1] is None:
            continue

        rank = beam_search(agent, env, e1, r, e2)
        ranks.append(rank)

    return np.array(ranks)

