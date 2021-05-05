import gym
import numpy as np
import torch
import torch.nn.functional as F

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

        self.banned_relations = [None] * batch_size
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

        self.rs_coef = config['rs_coef']


    def reset(self):
        random_idxs = np.random.randint(len(self.allowed_triplets), size=self.batch_size)
        chosen_triplets = self.allowed_triplets[random_idxs]
        self.init_entities = chosen_triplets[:, 0]
        self.target_relations = chosen_triplets[:, 1]
        self.target_entities = chosen_triplets[:, 2]

        self.current_entities = self.init_entities

        self.banned_relations = list(zip(self.init_entities, self.target_relations))

        return self._get_ob()

    def reward(self, e1, r, pred_e2, e2):
        binary_reward = float(pred_e2 == e2)
        if binary_reward == 0 and self.emb_model is not None:
            device = next(self.emb_model.parameters()).device

            e1 = torch.tensor([e1], device=device)
            r = torch.tensor([r], device=device)
            # pred_e2 = torch.tensor(pred_e2, device=device)

            with torch.no_grad():
                e2_prob = F.softmax(self.emb_model(e1, r))[0][pred_e2]
            return self.rs_coef * e2_prob
        else:
            return binary_reward

    def step(self, actions):
        """
        actions: indicies of relations in relations list
        """
        rewards = np.zeros(self.batch_size) + 0.01
        dones = [False] * self.batch_size

        for i in range(self.batch_size):
            relation = self.relations[actions[i]]
            if relation == self.STOP_IDX:
                rewards[i] = self.reward(
                    self.init_entities[i],
                    self.target_relations[i],
                    self.current_entities[i],
                    self.target_entities[i]
                )
                dones[i] = True
            else:
                entities = self.get_entities(self.current_entities[i], relation)

                self.current_entities[i] = np.random.choice(entities)

        return self._get_ob(), rewards, np.array(dones), None

    def get_possible_actions(self, entities):
        """
        Return indicies of possible relations in relations list
        """
        possible_actions = []
        for i, e in enumerate(entities):
            relations = [self.STOP_IDX] + list(self._graph[e].keys() if e in self._graph else [])
            if self.banned_relations[0] is not None and self.banned_relations[i][0] == e:
                relations.remove(self.banned_relations[i][1])

            possible_actions.append(relations)

        return possible_actions

    def _get_ob(self):
        return self.current_entities, self.target_relations