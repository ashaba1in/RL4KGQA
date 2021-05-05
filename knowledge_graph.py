import pandas as pd
import numpy as np

from drive.MyDrive.Diploma.load_config import config


class KnowledgeGraph:
    def __init__(self, triplets_file):
        self.entities = []
        with open(config['entities_file']) as e_f:
            for e in e_f:
                self.entities.append(int(e.strip()))
        self.entities = np.array(self.entities)

        self.relations = []
        with open(config['relations_file']) as r_f:
            for r in r_f:
                self.relations.append(int(r.strip()))
        self.relations = np.array(self.relations)

        self._graph = {}
        self.triplets = []
        # adding direct relations
        with open(triplets_file) as t_f:
            for line in t_f:
                e1, r, e2 = [int(x) for x in line.strip().split('\t')]
                self.add_relation(e1, r, e2)
                self.triplets.append([e1, r, e2])
        self.triplets = np.array(self.triplets)

        # adding absent inverse relations
        if config['add_reverse_relations']:
            for e1 in list(self._graph.keys()):
                for r in self._graph[e1].keys():
                    for e2 in self._graph[e1][r]:
                        if e2 not in self._graph:
                            self.add_relation(e2, self.inv(r), e1)
                        else:
                            has_inv = False
                            for _r in self._graph[e2]:
                                if e1 in self._graph[e2][_r]:
                                    has_inv = True
                                    break
                            if not has_inv:
                                self.add_relation(e2, self.inv(r), e1)

    def add_relation(self, e1, r, e2):
        node_relations = self._graph.get(e1, {})

        node_relations[r] = node_relations.get(r, []) + [e2]

        self._graph[e1] = node_relations

    def get_entities(self, e, r):
        return self._graph[e][r]

    def inv(self, r):
        return r + 1

    def __getitem__(self, e):
        return self._graph.get(e)
