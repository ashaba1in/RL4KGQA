import gym
import numpy as np
import torch
import torch.nn.functional as F

import copy

from drive.MyDrive.Diploma.knowledge_graph import KnowledgeGraph
from drive.MyDrive.Diploma.load_config import config


class KGEnv(gym.Env, KnowledgeGraph):
    """
    This is an environment for knowledge graph.
    Agent starts at a random entity and has a goal to get
    to another entity with a given relation between them.
    Number of relations is fixed and less than 100.
    **STATE:**
    The state consists of a starting entity, relation and current entity
    [e_s, r, e].
    **ACTIONS:**
    The action is any relation available from current entity or STOP.
    After choosing a relation agent moves to a random entity
    that has such relation with current entity.
    **REWARDS:**
    Agent gets +1 reward when stops at a correct entity and 0 otherwise.
    """

    def __init__(self, triplets_file, batch_size=1, train=True, emb_model=None):
        super().__init__(triplets_file)

        self.STOP_IDX = self.relations[-1]
        self.entities_num = len(self.entities)
        self.relations_num = len(self.relations)

        self.CLS = self.relations_num
        self.e_pad_idx = self.entities_num
        self.r_pad_idx = self.relations_num + 1

        self.init_entities = None
        self.target_relations = None
        self.target_entities = None
        self.current_entities = None

        self.banned_triplets = [None] * batch_size
        self.batch_size = batch_size

        if train:
            # choose only triplets with 2 or more relations from e1
            indices = []
            for i, (e1, _, _) in enumerate(self.triplets):
                if len(self._graph[e1]) > 1:
                    indices.append(i)

            self.allowed_triplets = np.array(self.triplets[indices])
        else:
            self.allowed_triplets = np.array(self.triplets)

        self.emb_model = emb_model
        if self.emb_model is not None:
            self.emb_model.eval()

        self.rs_coef = config['rs_coef'] if emb_model is not None else None


    def reset(self):
        random_idxs = np.random.randint(len(self.allowed_triplets), size=self.batch_size)
        chosen_triplets = self.allowed_triplets[random_idxs]
        self.init_entities = chosen_triplets[:, 0]
        self.target_relations = chosen_triplets[:, 1]
        self.target_entities = chosen_triplets[:, 2]

        self.current_entities = copy.deepcopy(self.init_entities)

        self.banned_triplets = list(zip(self.init_entities, self.target_relations, self.target_entities))

        return self._get_ob()

    def reward(self, e1, r_q, pred_e2, e2):
        binary_reward = float(pred_e2 in self.get_entities(e1, r_q))
        if binary_reward == 0 and self.emb_model is not None:
            device = next(self.emb_model.parameters()).device

            e1 = torch.tensor([e1], device=device)
            r = torch.tensor([r_q], device=device)

            with torch.no_grad():
                e2_prob = torch.sigmoid(self.emb_model(e1, r))[0][pred_e2]
            return self.rs_coef * e2_prob
        else:
            return binary_reward

    def step(self, actions, trace_idxs=None):
        if trace_idxs is None:
            trace_idxs = range(self.batch_size)

        rewards = np.zeros(self.batch_size)
        dones = np.ones(self.batch_size).astype(bool)  # all True
        dones[trace_idxs] = False

        assert len(trace_idxs) == len(actions)

        for i, action in zip(trace_idxs, actions):
            self.get_entities(self.init_entities[i], self.target_relations[i])
            if config['only_relations']:
                relation = action
            else:
                relation, entity = action

            if relation == self.STOP_IDX:
                rewards[i] = self.reward(
                    self.init_entities[i],
                    self.target_relations[i],
                    self.current_entities[i],
                    self.target_entities[i]
                )
                if config['static_relation'] == 'stop':
                    dones[i] = True
            else:
                if config['only_relations']:
                    entities = self.get_entities(self.current_entities[i], relation)
                    self.current_entities[i] = np.random.choice(entities)
                else:
                    self.current_entities[i] = entity

        return self._get_ob(), rewards, dones, None

    def get_possible_actions(self, entities, indices=None, ban=True):
        if indices is None:
            indices = range(len(entities))

        assert len(indices) == len(entities)

        possible_actions = []
        for i, e in zip(indices, entities):
            if config['only_relations']:
                relations = [self.STOP_IDX] + list(self._graph[e].keys() if e in self._graph else [])
                if ban and self.banned_triplets[0] is not None:  # check if we need to ban anything 
                    if self.banned_triplets[i][0] == e:
                        relations.remove(self.banned_triplets[i][1])

                possible_actions.append(relations)
            else:
                actions = []
                if e in self._graph:
                    actions = [(item[0], e_next) for item in self._graph[e].items() for e_next in item[1]]
                actions = [(self.STOP_IDX, e)] + actions
                if ban and self.banned_triplets[0] is not None:  # check if we need to ban anything
                    if self.banned_triplets[i][0] == e:
                        for e2 in self.get_entities(e, self.banned_triplets[i][1]):
                            actions.remove((self.banned_triplets[i][1], e2))

                possible_actions.append(list(zip(*actions)))

        return possible_actions

    def _get_ob(self):
        return self.current_entities, self.target_relations