# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : grammar.py
# @Software: PyCharm
"""

import re
import sys
import copy
import json
import torch
import numpy as np
import itertools
from collections import deque, namedtuple

sys.path.append("..")
from utils.query_interface import KB_query
from utils.utils import get_inv_edge, formalize_time_constraint, expand_variable_in_filter, \
    revert_period_constraint, revert_period_constraint_in_conds, \
    date_pattern_0, date_pattern_1, date_pattern_2, date_pattern_3, \
    int_pattern, float_pattern, year_pattern, quotation_pattern, \
    is_entity, is_type, normalize_relation, tokenize_word_sentence, STOP_RELATIONS
from utils.query_interface import query_ent_name


Triple = namedtuple('Triple', 'start, end, edge')


class HeterogeneousGraph(object):
    """
    Graph data structure,
    vertices have different class label, edges have different class label
    """
    def __init__(self):
        self.vertices = []      # save the name of each vertex, (INT).
                                # The x-th added vertex is at the x-th position of the list
        self.edges = []         # save the name of each edge, (INT)
                                # The x-th added edge is at the x-th position of the list

        self.v_labels = dict()  # class label of each vertex
        self.e_labels = dict()  # class label of each edge

        self.triples = []

    def get_vertex_pairs(self, v1, v2, both_ends=True):
        if both_ends:
            vertex_pairs = [[v1, v2], [v2, v1]]
        else:
            vertex_pairs = [[v1, v2]]
        return vertex_pairs

    def remove_edge(self, v1, v2, both_ends=True):
        vertex_pairs = self.get_vertex_pairs(v1, v2, both_ends)
        triples = self.triples[:]
        for triple in triples:
            if [triple.start, triple.end] in vertex_pairs:
                self.triples.remove(triple)

    def add_vertex(self, v, v_label):
        """
        add one vertex to the graph
        @param v:           vertex index
        @param v_label:     vertex class label
        """
        if v not in self.vertices:
            self.vertices.append(v)
            self.v_labels[v] = v_label

    def add_edge(self, e, e_label, v1, v2, both_ends=True):
        """
        add one edge to the graph
        @param e:           edge index
        @param e_label:     edge class label
        @param v1:          start vertex of the edge
        @param v2:          end vertex of the edge
        @param both_ends:   two-ways edge
        """
        if e not in self.edges:
            self.edges.append(e)
            self.e_labels[e] = e_label
        self.triples.append(Triple(start=v1, end=v2, edge=e))
        if both_ends:
            inv_e = get_inv_edge(e)
            if inv_e not in self.edges:
                self.edges.append(inv_e)
                self.e_labels[inv_e] = e_label
            self.triples.append(Triple(start=v2, end=v1, edge=inv_e))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for triple in self.triples:
            neighbours[triple.start].add((triple.end, triple.edge))
        for vertex in neighbours.keys():
            neighbours[vertex] = list(neighbours[vertex])
            neighbours[vertex].sort()
        return neighbours

    def get_vertex_label(self, v):
        return self.v_labels[v]

    def get_edge_label(self, e):
        return self.e_labels[e]

    @property
    def vertex_number(self):
        return len(self.vertices)

    @property
    def edge_number(self):
        return len(self.edges)


OPS = ["av", "sv", "ae"]
V_CLASS_IDS = {"ans": 0, "var": 1, "ent": 2, "type": 3, "val": 4, "end": 5}
E_CLASS_IDS = {"agg+": 0, "agg-": 1, "cmp+": 2, "cmp-": 3, "ord+": 4, "ord-": 5, "rel+": 6, "rel-": 7}


class AbstractQueryGraph(HeterogeneousGraph):
    """
    AQG data structure
    """

    def __init__(self):
        super(AbstractQueryGraph,self).__init__()

        self.v_segments = dict()        # Record the sub-query index where each vertex belong to
        self.e_segments = dict()        # Record the sub-query index where each edge belong to

        self.v_instances = dict()        # Record the original name of each vertex, such as, "ns:m.0f2y0"
        self.e_instances = dict()        # Record the original name of each edge, such as, "ns:film.performance.actor"

        self.pred_v_copy_labels = []
        self.pred_e_copy_labels = []
        self.pred_segment_labels = []


        self.score = 0.           # global probability of the operation sequence
        self.v_score = 0.
        self.e_score = 0.

    def init_state(self):
        self.vertices = []
        self.edges = []

        self.triples = []

        self.v_labels = dict()
        self.e_labels = dict()

        self.v_instances = dict()
        self.e_instances = dict()

        self.v_segments = dict()
        self.e_segments = dict()

        self.v_add_history = []     # The history of the vertex index in add vertex operation.
        self.e_add_history = []     # The history of the edge index in add edge operation.

        self.pred_obj_labels = []
        self.pred_v_copy_labels = []
        self.pred_e_copy_labels = []
        self.pred_segment_labels = []

        self.pred_v_instance_labels = []
        self.pred_e_instance_labels = []

        self.cur_segment = 0
        self.cur_v_add = -1
        self.cur_v_slc = -1
        self.op_idx = 0

        self.score = 0.
        self.v_score = 0.
        self.e_score = 0.

    def update_state(self, op, obj):
        """
        :param op:  { "av", "sv", "ae" }
        :param obj: object
        """
        if op == "av":
            v_class, copy_v, switch_segment = obj
            if v_class != V_CLASS_IDS["end"]:
                self.add_vertex_operation(v_class=v_class, switch_segment=switch_segment)
            self.pred_obj_labels.append(v_class)
            self.pred_v_copy_labels.append(copy_v)
            self.pred_segment_labels.append(switch_segment)

        elif op == "sv":
            self.select_vertex_operation(obj)
            self.pred_obj_labels.append(obj)

        elif op == "ae":
            e_class, e_copy = obj
            self.add_edge_operation(e_class=e_class, e_copy=e_copy)
            self.pred_obj_labels.append(e_class)
            self.pred_e_copy_labels.append(e_copy)

        else:
            raise ValueError("Operation \"{}\" is not defined !".format(op))

        self.op_idx += 1

    def get_state(self):

        vertices = [x for x in self.vertices]
        v_classes = {k: v for k, v in self.v_labels.items()}
        v_segments = {k: v for k, v in self.v_segments.items()}
        edges = [x for x in self.edges]
        e_classes = {k: v for k, v in self.e_labels.items()}
        e_segments = {k: v for k, v in self.e_segments.items()}
        triples = [x for x in self.triples]
        return vertices, v_classes, v_segments, edges, e_classes, e_segments, triples

    def show_state(self):
        """
        show current state of the aqg.
        :return:
        """
        V_CLASS = {v: k for k, v in V_CLASS_IDS.items()}
        E_CLASS = {e: k for k, e in E_CLASS_IDS.items()}

        vertices = [x for x in self.vertices]
        v_classes = {k:V_CLASS[v] for k, v in self.v_labels.items()}
        v_segments = {k:v for k, v in self.v_segments.items()}

        edges = [x for x in self.edges]
        e_classes = {k:E_CLASS[v] for k, v in self.e_labels.items()}
        e_segments = {k: v for k, v in self.e_segments.items()}

        triples = [[s, p, o] for s, o, p in self.triples]

        print("---------------------AQG-------------------------")
        print("--score:", self.score)
        print("--v_score:", self.v_score)
        print("--e_score:", self.e_score)
        print()
        print("--vertices:", vertices)
        print("--v_classes:", v_classes)
        print("--v_instances:", self.v_instances)
        print("--v_segments:", v_segments)
        print("--v_copy_labels:", self.pred_v_copy_labels)
        print("--segment_labels:", self.pred_segment_labels)
        print()
        print("--edges:", edges)
        print("--e_classes:", e_classes)
        print("--e_instances:", self.e_instances)
        print("--e_segments:", e_segments)
        print("--e_copy_labels:", self.pred_e_copy_labels)
        print()
        print("--triples:")
        for s, p, o in triples:
            print("  {}({}):{}   {}({}):{}   {}({}):{}".format(s, v_classes[s], v_segments[s],
                                                               p, e_classes[p], e_segments[p],
                                                               o, v_classes[o], v_segments[o]))
        print()

    def add_vertex_operation(self, v_class, switch_segment=False):
        """
        Add a new vertex into the AQG.
        @param v_class:         class of the new added vertex
        @param switch_segment:  True or False, whether enter a new segment (a new subquery)
        """
        self.cur_v_add = len(self.vertices)
        self.add_vertex(v=self.cur_v_add, v_label=v_class)
        if switch_segment == 1:
            self.cur_segment += 1
        self.v_segments[self.cur_v_add] = self.cur_segment
        self.v_add_history.append(self.cur_v_add)

    def select_vertex_operation(self, v):
        self.cur_v_slc = v

    def add_edge_operation(self, e_class, e_copy=-1):
        if e_copy != -1:
            assert e_copy in self.edges
            cur_e_add = e_copy
        else:
            cur_e_add = len(self.edges)
        cur_e_add_inv = get_inv_edge(cur_e_add)
        self.e_add_history.append(cur_e_add)

        if e_class % 2 == 0:
            # v_slc --> v_add
            self.add_edge(e=cur_e_add, e_label=e_class,
                          v1=self.cur_v_slc, v2=self.cur_v_add, both_ends=False)
            self.add_edge(e=cur_e_add_inv, e_label=e_class + 1,
                          v1=self.cur_v_add, v2=self.cur_v_slc, both_ends=False)    # add inverse edge for graph transformer
        else:
            # v_add --> v_slc
            self.add_edge(e=cur_e_add, e_label=e_class,
                          v1=self.cur_v_slc, v2=self.cur_v_add, both_ends=False)
            self.add_edge(e=cur_e_add_inv, e_label=e_class - 1,
                          v1=self.cur_v_add, v2=self.cur_v_slc, both_ends=False)    # add inverse edge for graph transformer

        self.e_segments[cur_e_add] = self.cur_segment
        self.e_segments[cur_e_add_inv] = self.cur_segment

    def is_equal(self, another_aqg):
        """
        check whether two aqg are identical
        """
        assert type(another_aqg) == AbstractQueryGraph

        if len(self.vertices) != len(another_aqg.vertices):
            return False, False

        # if len(self.edges) != len(another_aqg.edges):
        #     return False, False

        # check labels of vertices
        v_labels1 = [self.v_labels[x] for x in self.vertices]
        v_labels2 = [another_aqg.v_labels[x] for x in another_aqg.vertices]
        v_labels1 = " ".join([str(x) for x in sorted(v_labels1)])
        v_labels2 = " ".join([str(x) for x in sorted(v_labels2)])
        if v_labels1 != v_labels2:
            return False, False

        # # check labels of edges
        # e_labels1 = [self.e_labels[x] for x in self.edges]
        # e_labels2 = [another_aqg.e_labels[x] for x in another_aqg.edges]
        # e_labels1 = " ".join([str(x) for x in sorted(e_labels1)])
        # e_labels2 = " ".join([str(x) for x in sorted(e_labels2)])
        # if e_labels1 != e_labels2:
        #     return False, False

        # check index of subgraph
        v_segments1 = [self.v_segments[x] for x in self.vertices]
        v_segments2 = [another_aqg.v_segments[x] for x in another_aqg.vertices]
        v_segments1 = " ".join([str(x) for x in sorted(v_segments1)])
        v_segments2 = " ".join([str(x) for x in sorted(v_segments2)])
        if v_segments1 != v_segments2:
            return False, False

        triples1 = [[self.v_labels[t[0]], self.v_labels[t[1]], self.e_labels[t[2]]] for t in self.triples]
        triples2 = [[another_aqg.v_labels[t[0]], another_aqg.v_labels[t[1]], another_aqg.e_labels[t[2]]] for t in
                    another_aqg.triples]
        triples1 = ";".join(sorted([" ".join([str(x) for x in t]) for t in triples1]))
        triples2 = ";".join(sorted([" ".join([str(x) for x in t]) for t in triples2]))
        if triples1 != triples2:
            return False, False

        if self.vertex_number >= 8:
            instance_correct = True

            # check instances of vertices
            v_instances1 = [self.v_instances[x][-1] for x in self.vertices
                            if self.v_labels[x] != V_CLASS_IDS["ans"] and self.v_labels[x] != V_CLASS_IDS["var"]]
            v_instances2 = [another_aqg.v_instances[x][-1] for x in another_aqg.vertices
                            if
                            another_aqg.v_labels[x] != V_CLASS_IDS["ans"] and another_aqg.v_labels[x] != V_CLASS_IDS["var"]]
            v_instances1 = " ".join([str(x) for x in sorted(v_instances1)])
            v_instances2 = " ".join([str(x) for x in sorted(v_instances2)])
            if v_instances1 != v_instances2:
                instance_correct = False

            # check instances of edges
            e_instances1 = [self.e_instances[x][-1] for x in self.edges]
            e_instances2 = [another_aqg.e_instances[x][-1] for x in another_aqg.edges]
            e_instances1 = " ".join([str(x) for x in sorted(e_instances1)])
            e_instances2 = " ".join([str(x) for x in sorted(e_instances2)])
            if e_instances1 != e_instances2:
                instance_correct = False
            return True, instance_correct

        vertices1 = [v for v in self.vertices]
        vertex_idx1 = {v: i for i, v in enumerate(vertices1)}
        vertex_labels1 = [self.v_labels[v] for v in self.vertices]
        vertex_segments_labels1 = [self.v_segments[x] for x in self.vertices]
        vertex_instances_labels1 = [self.v_instances[x][-1]
                                    if self.v_labels[x] != V_CLASS_IDS["ans"] and self.v_labels[x] != V_CLASS_IDS["var"] else "var"
                                    for x in self.vertices]
        adj1 = np.full((len(vertices1), len(vertices1)), -1)
        for v1, v2, e in self.triples:
            adj1[vertex_idx1[v1]][vertex_idx1[v2]] = e
        adj_e_idx_flat1 = []
        adj_e_label_flat1 = []
        adj_e_instance_flat1 = []
        e_idx1 = dict()
        for e in adj1.flatten():
            if e == -1:
                adj_e_idx_flat1.append(str(-1))
                adj_e_label_flat1.append(str(-1))
                adj_e_instance_flat1.append(str(-1))
            else:
                if e not in e_idx1:
                    e_idx1[e] = len(e_idx1)
                adj_e_idx_flat1.append(str(e_idx1[e]))
                adj_e_label_flat1.append(str(self.e_labels[e]))
                if e in self.e_instances:
                    e_ins_1 = normalize_relation(self.e_instances[e][-1])
                    adj_e_instance_flat1.append(e_ins_1)
                else:
                    adj_e_instance_flat1.append("NONE")
        adj_e_idx_flat1 = " ".join(adj_e_idx_flat1)
        adj_e_label_flat1 = " ".join(adj_e_label_flat1)
        adj_e_instance_flat1 = " ".join(adj_e_instance_flat1)

        vertices2 = [v for v in another_aqg.vertices]

        structure_flag = False

        # Enumerate Vertex Permutations
        for perm in itertools.permutations([i for i in range(len(vertices2))], len(vertices2)):
            vertex_idx2 = {v: perm[i] for i, v in enumerate(vertices2)}
            vertex_labels2 = [0 for _ in range(len(vertices2))]
            vertex_segments_labels2 = [0 for _ in range(len(vertices2))]
            vertex_instances_labels2 = ["var" for _ in range(len(vertices2))]
            for v in vertices2:
                vertex_labels2[vertex_idx2[v]] = another_aqg.v_labels[v]
                vertex_segments_labels2[vertex_idx2[v]] = another_aqg.v_segments[v]
                if another_aqg.v_labels[v] not in [V_CLASS_IDS["ans"], V_CLASS_IDS["var"]]:
                    vertex_instances_labels2[vertex_idx2[v]] = another_aqg.v_instances[v][-1]

            # print(vertex_idx2)
            # print(vertex_labels1)
            # print(vertex_labels2)
            # print(vertex_segments_labels1)
            # print(vertex_segments_labels2)
            # print(vertex_instances_labels1)
            # print(vertex_instances_labels2)
            # print()

            # check class of vertex
            if " ".join([str(x) for x in vertex_labels1]) != " ".join([str(x) for x in vertex_labels2]):
                continue

            # check subquery of vertex
            if " ".join([str(x) for x in vertex_segments_labels1]) != " ".join([str(x) for x in vertex_segments_labels2]):
                continue

            adj2 = np.full((len(vertices2), len(vertices2)), -1)
            for v1, v2, e in another_aqg.triples:
                adj2[vertex_idx2[v1]][vertex_idx2[v2]] = e
            adj_e_idx_flat2 = []
            adj_e_label_flat2 = []
            adj_e_instance_flat2 = []
            e_idx2 = dict()
            for e in adj2.flatten():
                if e == -1:
                    adj_e_idx_flat2.append(str(-1))
                    adj_e_label_flat2.append(str(-1))
                    adj_e_instance_flat2.append(str(-1))
                else:
                    if e not in e_idx2:
                        e_idx2[e] = len(e_idx2)
                    adj_e_idx_flat2.append(str(e_idx2[e]))
                    adj_e_label_flat2.append(str(another_aqg.e_labels[e]))
                    e_ins_2 = normalize_relation(another_aqg.e_instances[e][-1])
                    adj_e_instance_flat2.append(e_ins_2)
            adj_e_idx_flat2 = " ".join([str(x) for x in adj_e_idx_flat2])
            adj_e_label_flat2 = " ".join([str(x) for x in adj_e_label_flat2])
            adj_e_instance_flat2 = " ".join([x for x in adj_e_instance_flat2])

            # print(adj_e_label_flat1 == adj_e_label_flat2)
            # print(adj1)
            # print(adj2)
            # print(adj_e_idx_flat1 == adj_e_idx_flat2)
            # print(adj_e_instance_flat1)
            # print(adj_e_instance_flat2)
            # print()

            if adj_e_label_flat1 == adj_e_label_flat2:
                structure_flag = True
                if adj_e_instance_flat1 == adj_e_instance_flat2 \
                        and " ".join(vertex_instances_labels1) == " ".join(vertex_instances_labels2):
                    return True, True
                continue
        return structure_flag, False

    def is_structure_equal(self, another_aqg):
        """
        check whether two aqg are identical
        """
        assert type(another_aqg) == AbstractQueryGraph

        if len(self.vertices) != len(another_aqg.vertices):
            return False

        # if len(self.edges) != len(another_aqg.edges):
        #     return False, False

        # check labels of vertices
        v_labels1 = [self.v_labels[x] for x in self.vertices]
        v_labels2 = [another_aqg.v_labels[x] for x in another_aqg.vertices]
        v_labels1 = " ".join([str(x) for x in sorted(v_labels1)])
        v_labels2 = " ".join([str(x) for x in sorted(v_labels2)])
        if v_labels1 != v_labels2:
            return False

        # # check labels of edges
        # e_labels1 = [self.e_labels[x] for x in self.edges]
        # e_labels2 = [another_aqg.e_labels[x] for x in another_aqg.edges]
        # e_labels1 = " ".join([str(x) for x in sorted(e_labels1)])
        # e_labels2 = " ".join([str(x) for x in sorted(e_labels2)])
        # if e_labels1 != e_labels2:
        #     return False, False

        # check index of subgraph
        v_segments1 = [self.v_segments[x] for x in self.vertices]
        v_segments2 = [another_aqg.v_segments[x] for x in another_aqg.vertices]
        v_segments1 = " ".join([str(x) for x in sorted(v_segments1)])
        v_segments2 = " ".join([str(x) for x in sorted(v_segments2)])
        if v_segments1 != v_segments2:
            return False

        triples1 = [[self.v_labels[t[0]], self.v_labels[t[1]], self.e_labels[t[2]]] for t in self.triples]
        triples2 = [[another_aqg.v_labels[t[0]], another_aqg.v_labels[t[1]], another_aqg.e_labels[t[2]]] for t in
                    another_aqg.triples]
        triples1 = ";".join(sorted([" ".join([str(x) for x in t]) for t in triples1]))
        triples2 = ";".join(sorted([" ".join([str(x) for x in t]) for t in triples2]))
        if triples1 != triples2:
            return False

        if self.vertex_number >= 8:
            return True

        vertices1 = [v for v in self.vertices]
        vertex_idx1 = {v: i for i, v in enumerate(vertices1)}
        vertex_labels1 = [self.v_labels[v] for v in self.vertices]
        vertex_segments_labels1 = [self.v_segments[x] for x in self.vertices]
        adj1 = np.full((len(vertices1), len(vertices1)), -1)
        for v1, v2, e in self.triples:
            adj1[vertex_idx1[v1]][vertex_idx1[v2]] = e
        adj_e_idx_flat1 = []
        adj_e_label_flat1 = []
        e_idx1 = dict()
        for e in adj1.flatten():
            if e == -1:
                adj_e_idx_flat1.append(str(-1))
                adj_e_label_flat1.append(str(-1))
            else:
                if e not in e_idx1:
                    e_idx1[e] = len(e_idx1)
                adj_e_idx_flat1.append(str(e_idx1[e]))
                adj_e_label_flat1.append(str(self.e_labels[e]))
        adj_e_idx_flat1 = " ".join(adj_e_idx_flat1)
        adj_e_label_flat1 = " ".join(adj_e_label_flat1)

        vertices2 = [v for v in another_aqg.vertices]

        structure_flag = False

        # Enumerate Vertex Permutations
        for perm in itertools.permutations([i for i in range(len(vertices2))], len(vertices2)):
            vertex_idx2 = {v: perm[i] for i, v in enumerate(vertices2)}
            vertex_labels2 = [0 for _ in range(len(vertices2))]
            vertex_segments_labels2 = [0 for _ in range(len(vertices2))]
            for v in vertices2:
                vertex_labels2[vertex_idx2[v]] = another_aqg.v_labels[v]
                vertex_segments_labels2[vertex_idx2[v]] = another_aqg.v_segments[v]

            # print(vertex_idx2)
            # print(vertex_labels1)
            # print(vertex_labels2)
            # print(vertex_segments_labels1)
            # print(vertex_segments_labels2)
            # print(vertex_instances_labels1)
            # print(vertex_instances_labels2)
            # print()

            # check class of vertex
            if " ".join([str(x) for x in vertex_labels1]) != " ".join([str(x) for x in vertex_labels2]):
                continue

            # check subquery of vertex
            if " ".join([str(x) for x in vertex_segments_labels1]) != " ".join([str(x) for x in vertex_segments_labels2]):
                continue

            adj2 = np.full((len(vertices2), len(vertices2)), -1)
            for v1, v2, e in another_aqg.triples:
                adj2[vertex_idx2[v1]][vertex_idx2[v2]] = e
            adj_e_idx_flat2 = []
            adj_e_label_flat2 = []
            e_idx2 = dict()
            for e in adj2.flatten():
                if e == -1:
                    adj_e_idx_flat2.append(str(-1))
                    adj_e_label_flat2.append(str(-1))
                else:
                    if e not in e_idx2:
                        e_idx2[e] = len(e_idx2)
                    adj_e_idx_flat2.append(str(e_idx2[e]))
                    adj_e_label_flat2.append(str(another_aqg.e_labels[e]))
            adj_e_idx_flat2 = " ".join([str(x) for x in adj_e_idx_flat2])
            adj_e_label_flat2 = " ".join([str(x) for x in adj_e_label_flat2])

            # print(adj_e_label_flat1 == adj_e_label_flat2)
            # print(adj1)
            # print(adj2)
            # print(adj_e_idx_flat1 == adj_e_idx_flat2)
            # print(adj_e_instance_flat1)
            # print(adj_e_instance_flat2)
            # print()

            if adj_e_label_flat1 == adj_e_label_flat2:
                structure_flag = True
                continue
        return structure_flag

    def to_temporary_structure(self):
        _aqg = copy.deepcopy(self)

        structures = []

        def _find(x):
            if x == pa[x]:
                return x
            pa[x] = _find(pa[x])
            return pa[x]

        def _union(x, y):
            px, py = _find(x), _find(y)
            if px != py:
                pa[px] = py

        n_segment = len(set(self.v_segments.values()))
        for seg in range(n_segment - 1, -1, -1):
            seg_conds = []

            # get conditions, filters and orders
            for i, triple in enumerate(self.triples):
                s, o, p = triple
                if self.v_segments[s] != seg or self.v_segments[o] != seg:
                    continue

                # Only judge the conditions triples
                if self.e_labels[p] in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                    seg_conds.append([s, p, o])

            pa = {}
            seg_v = list(set(sum(([t[0], t[2]] for t in seg_conds), [])))
            for v in seg_v:
                pa[v] = v
            for s, p, o in seg_conds:
                _union(s, o)
            pa_set = {_find(v) for v in seg_v}

            for pv in pa_set:
                tmp_aqg = AbstractQueryGraph()
                v_set = set()
                triples = [[s, p, o] for s, p, o in seg_conds if _find(s) == pv]
                for s, p, o in triples:
                    v_set.add(s)
                    v_set.add(o)
                for v in v_set:
                    v_class = _aqg.get_vertex_label(v)
                    tmp_aqg.add_vertex(v, v_class if v_class != V_CLASS_IDS["ans"] else V_CLASS_IDS["var"])
                    tmp_aqg.set_vertex_segment(v, 0)
                    if v in _aqg.v_instances:
                        tmp_aqg.set_vertex_instance(v, _aqg.get_vertex_instance(v))
                for s, p, o in triples:
                    tmp_aqg.add_edge(p, _aqg.get_edge_label(p), s, o, both_ends=False)
                    tmp_aqg.set_edge_segment(p, 0)
                    tmp_aqg.set_edge_instance(p, _aqg.get_edge_instance(p))

                structures.append(tmp_aqg)

        return structures

    def to_temporary_sparql_query(self, kb, select_relation=False):

        v_rename_mapping = dict()
        for v in self.vertices:
            if self.v_labels[v] == V_CLASS_IDS["ans"]:
                v_rename_mapping[v] = "?x" + "_" + str(self.v_segments[v])
            elif self.v_labels[v] == V_CLASS_IDS["var"]:
                v_rename_mapping[v] = "?v" + "_" + str(v) + "_" + str(self.v_segments[v])
            else:
                assert v in self.v_instances
                v_rename_mapping[v] = self.v_instances[v][-1]

        e_rename_mapping = dict()
        for e in self.edges:
            if e in self.e_instances:
                e_rename_mapping[e] = self.e_instances[e][-1]
            else:
                e_rename_mapping[e] = "?e" + "_" + str(e) + "_" + str(self.e_segments[e])

        queries = []

        n_segment = len(set(self.v_segments.values()))
        for seg in range(n_segment - 1, -1, -1):
            seg_sels = []
            seg_conds = []
            period_vars = []

            # get conditions, filters and orders
            for i, triple in enumerate(self.triples):
                if i % 2 == 1: continue
                s, o, p = triple
                if seg > 0 and (self.v_segments[s] != seg or self.v_segments[o] != seg):
                    continue

                if seg == 0 and (self.v_segments[s] != seg and self.v_segments[o] != seg):
                    continue

                # Only judge the conditions triples
                if self.e_labels[p] == E_CLASS_IDS["rel+"]:
                    if p in self.e_instances and "$$$" in self.e_instances[p][-1]:
                        p_from = self.e_instances[p][-1].split("$$$")[0]
                        p_to = ".".join(p_from.split(".")[:-1]) + "." + self.e_instances[p][-1].split("$$$")[-1]

                        flag = True
                        for j, (ss, oo, pp) in enumerate(self.triples):
                            if j == i: continue
                            if v_rename_mapping[o] == v_rename_mapping[oo] or v_rename_mapping[o] == v_rename_mapping[ss]:
                                flag = False
                                break
                        # can not split "from$$$to" or "start_date$$$end_date"
                        # so, the "from$$$to" or "start_date$$$end_date" should not be taken as the edge instance
                        if not flag or "?x" in v_rename_mapping[o]:
                            return []
                        seg_conds.append([v_rename_mapping[s], p_from, v_rename_mapping[o] + "_from"])
                        seg_conds.append([v_rename_mapping[s], p_to, v_rename_mapping[o] + "_to"])
                    else:
                        if kb == "dbpedia" and self.v_labels[o] == V_CLASS_IDS["type"]:
                            seg_conds.append([v_rename_mapping[s],
                                              "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                                              v_rename_mapping[o]])
                        else:
                            seg_conds.append([v_rename_mapping[s], e_rename_mapping[p], v_rename_mapping[o]])

                    if e_rename_mapping[p][0] == "?":
                        if kb != "dbpedia" or self.v_labels[o] != V_CLASS_IDS["type"]:
                            seg_sels.append(e_rename_mapping[p])

                elif self.e_labels[p] == E_CLASS_IDS["rel-"]:
                    if p in self.e_instances and "$$$" in self.e_instances[p][-1]:
                        p_from = self.e_instances[p][-1].split("$$$")[0]
                        p_to = ".".join(p_from.split(".")[:-1]) + "." + self.e_instances[p][-1].split("$$$")[-1]
                        flag = True
                        for j, (ss, oo, pp) in enumerate(self.triples):
                            if j == i: continue
                            if v_rename_mapping[s] == v_rename_mapping[oo] or v_rename_mapping[s] == v_rename_mapping[ss]:
                                flag = False
                                break
                        # can not split "from$$$to" or "start_date$$$end_date"
                        # so, the "from$$$to" or "start_date$$$end_date" should not be taken as the edge instance
                        if not flag or "?x" in v_rename_mapping[s]:
                            return []

                        seg_conds.append([v_rename_mapping[o], p_from, v_rename_mapping[s] + "_from"])
                        seg_conds.append([v_rename_mapping[o], p_to, v_rename_mapping[s] + "_to"])
                    else:
                        if kb == "dbpedia" and self.v_labels[s] == V_CLASS_IDS["type"]:
                            seg_conds.append([v_rename_mapping[o],
                                              "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                                              v_rename_mapping[s]])
                        else:
                            seg_conds.append([v_rename_mapping[o], e_rename_mapping[p], v_rename_mapping[s]])

                    if e_rename_mapping[p][0] == "?":
                        if kb != "dbpedia" or self.v_labels[s] != V_CLASS_IDS["type"]:
                            seg_sels.append(e_rename_mapping[p])

                elif self.e_labels[p] == E_CLASS_IDS["cmp+"] and "$$$" in v_rename_mapping[o]:
                    period_vars.append(v_rename_mapping[s])

                elif self.e_labels[p] == E_CLASS_IDS["cmp-"] and "$$$" in v_rename_mapping[s]:
                    period_vars.append(v_rename_mapping[o])

            saved_seg_conds = [x for x in seg_conds]
            seg_filters = expand_variable_in_filter(seg_conds)

            where_str = "\n".join([" ".join([x for x in one_cond]) + " ." for one_cond in seg_conds]) + "\n" \
                        + "\n".join(seg_filters)

            if kb == "freebase":
                prefix_str = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
                from_str = ""
            else:
                prefix_str = ""
                # from_str = "FROM <http://dbpedia.org/>\n"
                from_str = "FROM <dbpedia>\n"

            if select_relation:
                seg_sels_str = " ".join(list(set(seg_sels)))
                sparql_query = prefix_str + \
                               "SELECT DISTINCT " + seg_sels_str + "\n" + \
                               from_str + \
                               "WHERE {\n" + where_str + "}"
            else:
                sparql_query = prefix_str + \
                               "ASK\n" + \
                               from_str + \
                               "WHERE {\n" + where_str + "}"

            queries.append(sparql_query)

            if kb == "freebase" and period_vars:
                new_seg_sels = []
                new_seg_conds = []
                for s, p, o in saved_seg_conds:
                    if s not in period_vars and o not in period_vars:
                        new_seg_conds.append([s, p, o])
                        if p[0] == "?" and p not in new_seg_sels:
                            new_seg_sels.append(p)
                new_seg_filters = expand_variable_in_filter(new_seg_conds)

                select_str_1 = " ".join(new_seg_sels) + "\n"
                where_str_1 = "\n".join([" ".join([x for x in one_cond]) + " ." for one_cond in new_seg_conds]) + "\n" \
                            + "\n".join(new_seg_filters)
                sparql_query_1 = prefix_str + \
                                 "ASK\n" + \
                                 from_str + \
                                 "WHERE {\n" + where_str_1 + "}"
                queries.append(sparql_query_1)
        return queries

    def to_ask_sparql_query_for_eg(self, kb, qid):

        v_rename_mapping = dict()
        for v in self.vertices:
            if self.v_labels[v] == V_CLASS_IDS["ans"]:
                v_rename_mapping[v] = "?x" + "_" + str(self.v_segments[v])
            elif self.v_labels[v] == V_CLASS_IDS["var"]:
                v_rename_mapping[v] = "?v" + "_" + str(v) + "_" + str(self.v_segments[v])
            else:
                assert v in self.v_instances
                v_rename_mapping[v] = self.v_instances[v][-1]

        e_rename_mapping = dict()
        for e in self.edges:
            if e in self.e_instances:
                e_rename_mapping[e] = self.e_instances[e][-1]
            else:
                e_rename_mapping[e] = "?e" + "_" + str(e) + "_" + str(self.e_segments[e])

        queries = []

        n_segment = len(set(self.v_segments.values()))
        for seg in range(n_segment - 1, -1, -1):
            seg_sels = []
            seg_conds = []
            period_vars = []

            # get conditions, filters and orders
            for i, triple in enumerate(self.triples):
                if i % 2 == 1: continue
                s, o, p = triple
                if seg > 0 and (self.v_segments[s] != seg or self.v_segments[o] != seg):
                    continue

                if seg == 0 and (self.v_segments[s] != seg and self.v_segments[o] != seg):
                    continue

                # Only judge the conditions triples
                if self.e_labels[p] == E_CLASS_IDS["rel+"]:
                    if p in self.e_instances and "$$$" in self.e_instances[p][-1]:
                        p_from = self.e_instances[p][-1].split("$$$")[0]
                        p_to = ".".join(p_from.split(".")[:-1]) + "." + self.e_instances[p][-1].split("$$$")[-1]

                        flag = True
                        for j, (ss, oo, pp) in enumerate(self.triples):
                            if j == i: continue
                            if v_rename_mapping[o] == v_rename_mapping[oo] or v_rename_mapping[o] == v_rename_mapping[ss]:
                                flag = False
                                break
                        # can not split "from$$$to" or "start_date$$$end_date"
                        # so, the "from$$$to" or "start_date$$$end_date" should not be taken as the edge instance
                        if not flag or "?x" in v_rename_mapping[o]:
                            return []
                        seg_conds.append([v_rename_mapping[s], p_from, v_rename_mapping[o] + "_from"])
                        seg_conds.append([v_rename_mapping[s], p_to, v_rename_mapping[o] + "_to"])
                    else:
                        if kb == "dbpedia" and self.v_labels[o] == V_CLASS_IDS["type"]:
                            seg_conds.append([v_rename_mapping[s],
                                              "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                                              v_rename_mapping[o]])
                        else:
                            seg_conds.append([v_rename_mapping[s], e_rename_mapping[p], v_rename_mapping[o]])

                    if e_rename_mapping[p][0] == "?":
                        if kb != "dbpedia" or self.v_labels[o] != V_CLASS_IDS["type"]:
                            seg_sels.append(e_rename_mapping[p])

                elif self.e_labels[p] == E_CLASS_IDS["rel-"]:
                    if p in self.e_instances and "$$$" in self.e_instances[p][-1]:
                        p_from = self.e_instances[p][-1].split("$$$")[0]
                        p_to = ".".join(p_from.split(".")[:-1]) + "." + self.e_instances[p][-1].split("$$$")[-1]
                        flag = True
                        for j, (ss, oo, pp) in enumerate(self.triples):
                            if j == i: continue
                            if v_rename_mapping[s] == v_rename_mapping[oo] or v_rename_mapping[s] == v_rename_mapping[ss]:
                                flag = False
                                break
                        # can not split "from$$$to" or "start_date$$$end_date"
                        # so, the "from$$$to" or "start_date$$$end_date" should not be taken as the edge instance
                        if not flag or "?x" in v_rename_mapping[s]:
                            return []

                        seg_conds.append([v_rename_mapping[o], p_from, v_rename_mapping[s] + "_from"])
                        seg_conds.append([v_rename_mapping[o], p_to, v_rename_mapping[s] + "_to"])
                    else:
                        if kb == "dbpedia" and self.v_labels[s] == V_CLASS_IDS["type"]:
                            seg_conds.append([v_rename_mapping[o],
                                              "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                                              v_rename_mapping[s]])
                        else:
                            seg_conds.append([v_rename_mapping[o], e_rename_mapping[p], v_rename_mapping[s]])

                    if e_rename_mapping[p][0] == "?":
                        if kb != "dbpedia" or self.v_labels[s] != V_CLASS_IDS["type"]:
                            seg_sels.append(e_rename_mapping[p])

                elif self.e_labels[p] == E_CLASS_IDS["cmp+"] and "$$$" in v_rename_mapping[o]:
                    period_vars.append(v_rename_mapping[s])

                elif self.e_labels[p] == E_CLASS_IDS["cmp-"] and "$$$" in v_rename_mapping[s]:
                    period_vars.append(v_rename_mapping[o])

            saved_seg_conds = [x for x in seg_conds]
            seg_filters = expand_variable_in_filter(seg_conds)

            if kb == "freebase":
                where_str = "\n".join([" ".join(["<" + x + ">" if x[0] != "?" else x for x in one_cond])
                                       + " ." for one_cond in seg_conds]) + "\n" \
                            + "\n".join(seg_filters)
            else:
                where_str = "\n".join([" ".join([x if x[0] != "?" else x for x in one_cond])
                                       + " ." for one_cond in seg_conds]) + "\n" \
                            + "\n".join(seg_filters)

            prefix_str = ""
            from_str = "FROM <" + qid + ">\n"

            sparql_query = prefix_str + \
                           "ASK\n" + \
                           from_str + \
                           "WHERE {\n" + where_str + "}"

            queries.append(sparql_query)

            if kb == "freebase" and period_vars:
                new_seg_sels = []
                new_seg_conds = []
                for s, p, o in saved_seg_conds:
                    if s not in period_vars and o not in period_vars:
                        new_seg_conds.append([s, p, o])
                        if p[0] == "?" and p not in new_seg_sels:
                            new_seg_sels.append(p)
                new_seg_filters = expand_variable_in_filter(new_seg_conds)

                where_str_1 = "\n".join([" ".join([x for x in one_cond]) + " ." for one_cond in new_seg_conds]) + "\n" \
                            + "\n".join(new_seg_filters)
                sparql_query_1 = prefix_str + \
                                 "ASK\n" + \
                                 from_str + \
                                 "WHERE {\n" + where_str_1 + "}"
                queries.append(sparql_query_1)
        return queries

    def to_final_sparql_query(self, kb):

        v_rename_mapping = dict()
        for v in self.vertices:
            if self.v_labels[v] == V_CLASS_IDS["ans"]:
                v_rename_mapping[v] = "?x" + "_" + str(self.v_segments[v])
            elif self.v_labels[v] == V_CLASS_IDS["var"]:
                v_rename_mapping[v] = "?v" + "_" + str(v) + "_" + str(self.v_segments[v])
            else:
                assert v in self.v_instances
                v_rename_mapping[v] = self.v_instances[v][-1]

        e_rename_mapping = dict()
        for e in self.edges:
            assert e in self.e_instances
            e_rename_mapping[e] = self.e_instances[e][-1]

        sub_sparql_list = []

        date_vars = []
        year_vars = []
        period_vars = []        # the variables of periods, $$$

        main_sparql = ""

        n_segment = len(set(self.v_segments.values()))
        for seg in range(n_segment - 1, -1, -1):
            seg_sels = []
            seg_aggs = []
            seg_conds = []
            seg_filters = []
            seg_filters_main = []    # filter in the main query.
            seg_ords = []

            # find select variable in current segment
            for i, triple in enumerate(self.triples):
                if i % 2 == 1: continue
                s, o, p = triple
                if seg == 0:
                    # final answer of the main-query
                    if self.v_labels[s] == V_CLASS_IDS["ans"]:
                        seg_sels.append(v_rename_mapping[s])
                        break
                    if self.v_labels[o] == V_CLASS_IDS["ans"]:
                        seg_sels.append(v_rename_mapping[o])
                        break
                else:
                    # the variable connect the sub-query with main-query
                    if self.v_segments[s] == seg and self.v_segments[o] < seg:
                        seg_sels.append(v_rename_mapping[s])
                        break
                    if self.v_segments[o] == seg and self.v_segments[s] < seg:
                        seg_sels.append(v_rename_mapping[o])
                        break

            # get conditions, filters and orders
            for i, triple in enumerate(self.triples):
                if i % 2 == 1: continue
                s, o, p = triple

                if seg > 0 and (self.v_segments[s] != seg or self.v_segments[o] != seg):
                    continue

                if seg == 0 and (self.v_segments[s] != seg and self.v_segments[o] != seg):
                    continue

                # Only judge the conditions triples
                if self.e_labels[p] == E_CLASS_IDS["rel+"]:
                    seg_conds.append([v_rename_mapping[s], e_rename_mapping[p], v_rename_mapping[o]])
                    if e_rename_mapping[p].split(".")[-1] in ["from", "to", "start_date", "end_date"]:
                        date_vars.append(v_rename_mapping[o])
                    if e_rename_mapping[p].split(".")[-1] in ["year", "founded"]:
                        year_vars.append(v_rename_mapping[o])

                elif self.e_labels[p] == E_CLASS_IDS["rel-"]:
                    seg_conds.append([v_rename_mapping[o], e_rename_mapping[p], v_rename_mapping[s]])
                    if e_rename_mapping[p].split(".")[-1] in ["from", "to", "start_date", "end_date"]:
                        date_vars.append(v_rename_mapping[s])
                    if e_rename_mapping[p].split(".")[-1] in ["year", "founded"]:
                        year_vars.append(v_rename_mapping[s])

                elif self.e_labels[p] == E_CLASS_IDS["agg+"]:
                    seg_aggs.append([v_rename_mapping[s], e_rename_mapping[p], v_rename_mapping[o]])
                elif self.e_labels[p] == E_CLASS_IDS["agg-"]:
                    seg_aggs.append([v_rename_mapping[o], e_rename_mapping[p], v_rename_mapping[s]])

                elif self.e_labels[p] == E_CLASS_IDS["ord+"]:
                    seg_ords.append([v_rename_mapping[s], e_rename_mapping[p], v_rename_mapping[o]])
                elif self.e_labels[p] == E_CLASS_IDS["ord-"]:
                    seg_ords.append([v_rename_mapping[o], e_rename_mapping[p], v_rename_mapping[s]])

                elif self.e_labels[p] == E_CLASS_IDS["cmp+"]:
                    if self.v_segments[s] != seg or self.v_segments[o] != seg:
                        # the filters that contains variable in different segments should be in the main-query
                        seg_filters_main.append([v_rename_mapping[s], e_rename_mapping[p], v_rename_mapping[o]])
                    else:
                        seg_filters.append([v_rename_mapping[s], e_rename_mapping[p], v_rename_mapping[o]])

                elif self.e_labels[p] == E_CLASS_IDS["cmp-"]:
                    if self.v_segments[s] != seg or self.v_segments[o] != seg:
                        seg_filters_main.append([v_rename_mapping[o], e_rename_mapping[p], v_rename_mapping[s]])
                    else:
                        seg_filters.append([v_rename_mapping[o], e_rename_mapping[p], v_rename_mapping[s]])

            seg_sels, seg_conds, seg_filters, seg_filters_main = revert_period_constraint(period_vars,
                                                                                          seg_sels, seg_conds,
                                                                                          seg_filters, seg_filters_main)

            new_filters = []
            new_filters += formalize_time_constraint(seg_conds, seg_filters)
            new_filters += expand_variable_in_filter(seg_conds)

            # Build components
            select_list = []
            where_list = []
            filter_list = []
            filter_list_main = []
            order_list = []

            select_count_flag = False
            select_ask_flag = False
            if len(seg_aggs) > 0:
                assert len(seg_sels) == 1
                assert len(seg_aggs) <= 2
                if len(seg_aggs) == 2 and seg_aggs[1][-1] == seg_sels[0]:
                    # There are nested queries
                    # SELECT (MAX(?count) AS ?maxCount)
                    #       WHERE {
                    #         SELECT ?x (COUNT(?x) as ?count)
                    #         WHERE {
                    #           ns:m.02_p0 ns:sports.sports_award_type.winners ?y .
                    #           ?y ns:sports.sports_award.award_winner ?x .
                    #         }
                    #       }
                    # seg_aggs: [[?x, COUNT, ?count], [?count, MAX, ?maxCount]]
                    #           ---> [[?count, MAX, ?maxCount], [?x, COUNT, ?count]]
                    # Low-level selects are in the back
                    seg_aggs[0], seg_aggs[1] = seg_aggs[1], seg_aggs[0]

                for v1, agg, v2 in seg_aggs:
                    if agg == "ASK":
                        select_list.append("ASK")
                        select_ask_flag = True
                        break

                    # [?count, MAX, ?maxCount] ---> "(MAX(?count) AS ?maxCount)"
                    if v2 == "?x_0":
                        select_list.append(agg + "(" + v1 + ")")
                        select_count_flag = True
                    else:
                        select_list.append(v1 + " (" + agg + "(" + v1 + ") AS " + v2 + ")")

                if select_count_flag or select_ask_flag:
                    # SELECT DISTINCT COUNT(?uri)
                    # WHERE {
                    # ?uri <http://dbpedia.org/property/network> <http://dbpedia.org/resource/Comedy_Central>  .
                    # ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/TelevisionShow>
                    # }
                    seg_aggs = []
            else:
                select_list.append(" ".join(seg_sels))

            for s, p, o in seg_conds:
                where_list.append(" ".join([s, p, o]) + " .\n")

            for v1, cmp, v2 in seg_filters:
                filter_list.append("FILTER (" + " ".join([v1, cmp, v2]) + ")\n")
            for f in new_filters:
                filter_list.append(f + "\n")

            for v, ord, num in seg_ords:
                if v in date_vars:
                    v_type = "xsd:datetime"
                elif v in year_vars:
                    v_type = ""
                else:
                    v_type = "xsd:float"
                order_list.append(ord + "(" + v_type + "(" + v + "))\nLIMIT " + num)

            # print(select_list)
            # print(where_list)
            # print(filter_list)
            # print(filter_list_main)
            # print(order_list)

            # Combine to the SPARQL string
            select_str = "SELECT DISTINCT " + select_list[-1] + "\n"     # Low-level selects are in the back
            where_str = "WHERE {\n" + "\n".join(where_list) + " " \
                                   + "\n".join(filter_list) + "\n}\n"
            if kb == "dbpedia":
                from_str = "FROM <dbpedia>\n"
            else:
                from_str = "\n"

            if order_list:
                order_str = "\nORDER BY " + order_list[0]
            else:
                order_str = ""

            sparql = select_str + from_str + where_str + order_str

            if len(seg_aggs) == 2:
                # build nested queries for aggregation on select.
                sparql = "SELECT DISTINCT " + select_list[0] + "\n" + \
                         from_str + \
                         "WHERE {\n" + sparql + "\n}" + \
                         order_str

            if seg > 0:
                sub_sparql_list.append("{\n" + sparql + "\n}")
            else:
                for v1, cmp, v2 in seg_filters_main:
                    filter_list_main.append("FILTER (" + " ".join([v1, cmp, v2]) + ")\n")

                if seg_aggs:
                    sparql = "SELECT DISTINCT ?x_0\n" + from_str + \
                             "WHERE {\n" + \
                             "{\n" + sparql + "\n} " + \
                             "".join(sub_sparql_list) + \
                             "".join(filter_list_main) + \
                             "}" + \
                             order_str
                else:
                    if select_ask_flag:
                        sparql = "ASK\n" + from_str + \
                                 "WHERE {\n" + \
                                 "".join(where_list) + \
                                 "".join(filter_list) + \
                                 "".join(sub_sparql_list) + \
                                 "".join(filter_list_main) + \
                                 "}" + \
                                 order_str
                    else:
                        sparql = select_str + from_str + \
                                 "WHERE {\n" + \
                                 "".join(where_list) + \
                                 "".join(filter_list) + \
                                 "".join(sub_sparql_list) + \
                                 "".join(filter_list_main) + \
                                 "}" + \
                                 order_str
                main_sparql = sparql

        if kb == "freebase":
            prefix = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
        else:
            prefix = ""

        return prefix + main_sparql

    def check_final_structure(self, instance_pool, dataset):
        if len(self.triples) == 0:
            return False

        cnt_pred_ent = 0
        cnt_pred_type = 0
        cnt_pred_var = 0
        for v, v_class in self.v_labels.items():
            if v_class == V_CLASS_IDS["ent"]:
                cnt_pred_ent += 1
            if v_class == V_CLASS_IDS["type"]:
                cnt_pred_type += 1
            if v_class == V_CLASS_IDS["var"]:
                cnt_pred_var += 1
        if cnt_pred_ent == 0:
            return False

        cnt_pred_rel = 0
        cnt_pred_agg = 0
        for e, e_class in self.e_labels.items():
            if e_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                cnt_pred_rel += 1
            if e_class in [E_CLASS_IDS["agg+"], E_CLASS_IDS["agg-"]]:
                cnt_pred_agg += 1
        if cnt_pred_rel == 0:
            return False

        if dataset == "lcq":
            if cnt_pred_rel // 2 >= 4:
                return False
            if cnt_pred_rel // 2 >= 3 and cnt_pred_ent + cnt_pred_type <= 1:
                return False

        if V_CLASS_IDS["ent"] not in instance_pool["vertex"]:
            cnt_cand_ent = 0
        else:
            cand_ents = [x[0] for x in instance_pool["vertex"][V_CLASS_IDS["ent"]]]
            cnt_cand_ent = len(cand_ents)
        if cnt_cand_ent != cnt_pred_ent:
            return False

        if V_CLASS_IDS["type"] not in instance_pool["vertex"]:
            cnt_cand_type = 0
        else:
            cand_types = [x[0] for x in instance_pool["vertex"][V_CLASS_IDS["type"]]]
            cnt_cand_type = len(cand_types)
        if cnt_cand_type < cnt_pred_type:
            return False

        if dataset == "lcq":
            if cnt_pred_agg > 0 and cnt_pred_var > 0:
                if cnt_cand_type > 0 and cnt_pred_type == 0:
                    return False
        return True

    def check_temporary_structure(self, dataset):

        has_var_or_type = False
        for v, v_class in self.v_labels.items():
            if v_class in [V_CLASS_IDS["type"], V_CLASS_IDS["var"]]:
                has_var_or_type = True
                break

        for i, (s, o, p) in enumerate(self.triples):
            # only handle edge with direction "+"
            if self.e_labels[p] % 2 == 1:
                continue
            s_class = self.v_labels[s]
            o_class = self.v_labels[o]
            p_class = self.e_labels[p]

            # V_CLASS_IDS = {"ans": 0, "var": 1, "ent": 2, "type": 3, "val": 4, "end": 5}
            # E_CLASS_IDS = {"agg+": 0, "agg-": 1, "cmp+": 2, "cmp-": 3, "ord+": 4, "ord-": 5, "rel+": 6, "rel-": 7}

            if p_class in [E_CLASS_IDS["agg+"], E_CLASS_IDS["agg-"]]:
                if dataset != "lcq":
                    if s_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] or o_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                        return False
                else:
                    if (s_class == V_CLASS_IDS["ent"] or o_class == V_CLASS_IDS["ent"]) and has_var_or_type:
                        return False

            if p_class in [E_CLASS_IDS["cmp+"], E_CLASS_IDS["cmp-"]]:
                if s_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] \
                        and (o_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] and o_class != V_CLASS_IDS["val"]):
                    return False
                if o_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] \
                        and (s_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] and s_class != V_CLASS_IDS["val"]):
                    return False

            if p_class == E_CLASS_IDS["ord+"]:
                if s_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] or o_class != V_CLASS_IDS["val"]:
                    return False
            if p_class == E_CLASS_IDS["ord-"]:
                if o_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] or s_class != V_CLASS_IDS["val"]:
                    return False

            if p_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                if dataset != "lcq" or (dataset == "lcq" and has_var_or_type):
                    if s_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] and o_class not in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                        return False

        return True

    def check_final_query_graph(self, kb):
        if kb == "dbpedia":
            for e, e_class in self.e_labels.items():
                if e_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                    e_name = self.get_edge_instance(e)[-1]
                    if e_name in STOP_RELATIONS:
                        return False
                    _cnt = len([c for c in e_name if c == "/"])
                    if _cnt > 4 and e_name != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                        return False

            type_v_true_name = None
            for v, v_class in self.v_labels.items():
                if v_class == V_CLASS_IDS["type"]:
                    v_name = self.get_vertex_instance(v)[-1]
                    type_v_true_name = get_type_true_name(v_name, kb=kb).lower()

            if type_v_true_name is not None:
                for e, e_class in self.e_labels.items():
                    if e_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                        e_name = self.get_edge_instance(e)[-1]
                        e_true_name = get_relation_true_name(e_name, kb=kb).lower()
                        e_true_name = " ".join(e_true_name.split(" ")[1:])
                        if type_v_true_name == e_true_name:
                            return False

                    if e_class in [E_CLASS_IDS["agg+"], E_CLASS_IDS["agg-"]]:
                        e_name = self.get_edge_instance(e)[-1]
                        if e_name == "ASK":
                            return False
        return True

    def normalize(self, dataset):

        v_rename_mapping = dict()
        for v in self.vertices:
            if self.v_labels[v] == V_CLASS_IDS["ans"]:
                v_rename_mapping[v] = "?x" + "_" + str(self.v_segments[v])
            elif self.v_labels[v] == V_CLASS_IDS["var"]:
                v_rename_mapping[v] = "?v" + "_" + str(v) + "_" + str(self.v_segments[v])
            else:
                assert v in self.v_instances
                v_rename_mapping[v] = self.v_instances[v][-1]

        e_rename_mapping = dict()
        for e in self.edges:
            assert e in self.e_instances
            e_rename_mapping[e] = self.e_instances[e][-1]

        if dataset == "lcq":
            pass
        else:
            date_vars = set()
            year_vars = set()
            period_vars = set()  # the variables of periods, $$$
            for i, (s, o, p) in enumerate(self.triples):
                # only handle edge with direction "+"
                if self.e_labels[p] == E_CLASS_IDS["cmp+"]:
                    if "$$$" in v_rename_mapping[o]:
                        period_vars.add(v_rename_mapping[s])
                        if e_rename_mapping[p] not in ["overlap", "during"]:
                            self.e_instances[p][-1] = "overlap"
                    elif "xsd:dateTime" in v_rename_mapping[o]:
                        res = re.search(date_pattern_1, v_rename_mapping[o])
                        if res:
                            date_vars.add(v_rename_mapping[s])
                        else:
                            year_vars.add(v_rename_mapping[s])

                elif self.e_labels[p] == E_CLASS_IDS["cmp-"]:
                    if "$$$" in v_rename_mapping[s]:
                        period_vars.add(v_rename_mapping[o])
                        if e_rename_mapping[p] not in ["overlap", "during"]:
                            self.e_instances[p][-1] = "overlap"
                    elif "xsd:dateTime" in v_rename_mapping[s]:
                        res = re.search(date_pattern_1, v_rename_mapping[s])
                        if res:
                            date_vars.add(v_rename_mapping[o])
                        else:
                            year_vars.add(v_rename_mapping[o])

            for i, (s, o, p) in enumerate(self.triples):
                if self.e_labels[p] == E_CLASS_IDS["rel+"]:
                    if v_rename_mapping[o] in period_vars and "$$$" not in e_rename_mapping[p]:
                        if e_rename_mapping[p].split(".")[-1] in ["from", "to"]:
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["from$$$to"])
                        elif e_rename_mapping[p].split(".")[-1] in ["start_date", "end_date"]:
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["start_date$$$end_date"])
                    elif (v_rename_mapping[o] in date_vars or v_rename_mapping[o] in year_vars) and "$$$" in e_rename_mapping[p]:
                        if e_rename_mapping[p].split(".")[-1] == "from$$$to":
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["from"])
                        elif e_rename_mapping[p].split(".")[-1] == "start_date$$$end_date":
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["start_date"])

                if self.e_labels[p] == E_CLASS_IDS["rel-"]:
                    if v_rename_mapping[s] in period_vars and "$$$" not in e_rename_mapping[p]:
                        if e_rename_mapping[p].split(".")[-1] in ["from", "to"]:
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["from$$$to"])
                        elif e_rename_mapping[p].split(".")[-1] in ["start_date", "end_date"]:
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["start_date$$$end_date"])
                    elif (v_rename_mapping[s] in date_vars or v_rename_mapping[s] in year_vars) and "$$$" in e_rename_mapping[p]:
                        if e_rename_mapping[p].split(".")[-1] == "from$$$to":
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["from"])
                        elif e_rename_mapping[p].split(".")[-1] == "start_date$$$end_date":
                            self.e_instances[p][-1] = ".".join(e_rename_mapping[p].split(".")[:-1] + ["start_date"])

    @property
    def cur_operation(self):
        if self.op_idx == 0:
            return 'av'
        else:
            return OPS[(self.op_idx - 1) % 3]

    def get_score(self):
        return self.score

    def update_score(self, score):
        self.score = score

    def get_v_score(self):
        return self.v_score

    def update_v_score(self, score):
        self.v_score = score

    def get_e_score(self):
        return self.e_score

    def update_e_score(self, score):
        self.e_score = score

    def set_vertex_segment(self, v, v_segment):
        self.v_segments[v] = v_segment

    def get_vertex_segment(self, v):
        return self.v_segments[v]

    def set_edge_segment(self, e, e_segment):
        self.e_segments[e] = e_segment

    def get_edge_segment(self, e):
        return self.e_segments[e]

    def set_vertex_instance(self, v, v_instance):
        """
        @param v:           v in self.vertices
        @param v_instance:  [instance_id, instance_name], "instance_id": the index of the instance in the pool
                                                          "instance_name": the name of the instance,
                                                          such as "ns:m.03d0l76"
        """
        self.v_instances[v] = v_instance

    def get_vertex_instance(self, v):
        return self.v_instances[v]

    def set_edge_instance(self, e, e_instance):
        """
        @param e:           e in self.edges
        @param e_instance:  [instance_id, instance_name], "instance_id": the index of the instance in the pool
                                                          "instance_name": the name of the instance,
                                                          such as "ns:location.mailing_address.state_province_region"
        """
        self.e_instances[e] = e_instance

    def get_edge_instance(self, e):
        return self.e_instances[e]

    def get_v_add_history(self, t):
        return self.v_add_history[t]

    def get_e_add_history(self, t):
        return self.e_add_history[t]


AGG_NAME = ["COUNT", "MAX", "MIN", "ASK"]
AGG_TRUE_NAME = ["count", "maximum", "minimum", "ask"]

CMP_NAME = ["=", "!=", ">", "<", ">=", "<=", "during", "overlap"]
CMP_TRUE_NAME = ["=", "!=", ">", "<", ">=", "<=", "during", "overlap"]

ORD_NAME = ["ASC", "DESC"]
ORD_TRUE_NAME = ["ascend", "descend"]

def cal_edge_matching_total_score(dataset, aqg, e_instance_literal_score):
    total_score = 0
    visit_edges = set()

    if dataset == "lcq":
        kb = "dbpedia"
    else:
        kb = "freebase"

    if dataset in ["lcq", "wsp"]:

        name_set = set()
        for i, e in enumerate(aqg.edges):
            if i % 2 == 1:
                continue
            if aqg.get_edge_label(e) not in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
                continue
            e_instance_name = aqg.get_edge_instance(e)[-1]
            e_instance_true_name = get_relation_true_name(e_instance_name, kb=kb)
            e_last_name = " ".join(e_instance_true_name.split(" ")[1:]).rstrip("s")
            e_name_toks = tokenize_word_sentence(e_last_name)
            if e_instance_true_name not in visit_edges:
                visit_edges.add(e_instance_true_name)

                common = False
                for _e_last_name in name_set:
                    _e_name_toks = tokenize_word_sentence(_e_last_name)
                    # print(set(e_name_toks) & set(_e_name_toks))
                    # print(set(e_name_toks) | set(_e_name_toks))
                    # print()
                    if len(set(e_name_toks) & set(_e_name_toks)) >= len(set(e_name_toks) | set(_e_name_toks)) // 2:
                        common = True
                        break
                if not common:
                    total_score += e_instance_literal_score[e_instance_name]
                    name_set.add(e_last_name)

                # if e_last_name not in name_set:
                #     total_score += e_instance_literal_score[e_instance_name]
                #     name_set.add(e_last_name)
        # if len(name_set) == len(aqg.edges) // 2 and total_score != 0:
        #     total_score += 0.5
    return total_score

def get_entity_true_name(ent_name, kb="freebase", kb_endpoint=None):
    if kb == "freebase":
        return query_ent_name(ent_name, kb_endpoint)
    else:
        return ent_name.strip("<").strip(">").split("/")[-1].replace("_", " ")

def get_type_true_name(type_name, kb="freebase"):
    if kb == "freebase":
        return type_name.replace("ns:", "").replace(".", " ").replace("_", " ")
    else:
        tmp_type_name = type_name.strip("<").strip(">").split("/")[-1]
        tokens = []
        last = 0
        for i, c in enumerate(tmp_type_name):
            if c.isupper():
                tokens.append(tmp_type_name[last: i])
                last = i
        tokens.append(tmp_type_name[last: len(tmp_type_name)])
        tokens = [x for x in tokens if x != ""]
        return " ".join(tokens)

def get_value_true_name(val_name, kb="freebase"):
    if kb == "freebase":
        if "xsd:dateTime" in val_name:
            if "$$$" in val_name:
                st, ed = val_name.split("$$$")
                st_date = st.replace("^^xsd:dateTime", "").strip("\"").replace("-", " ")
                ed_date = ed.replace("^^xsd:dateTime", "").strip("\"").replace("-", " ")
                val_true_name = "period : " + st_date + " - " + ed_date
            else:
                val_true_name = val_name.replace("^^xsd:dateTime", "").strip("\"").replace("-", " ")
                val_true_name = "date : " + val_true_name
        else:
            val_true_name = val_name.replace("\"@en", "").strip("\"")
        return val_true_name
    else:
        return val_name

def get_relation_last_name(rel_name, kb="freebase"):
    if kb == "freebase":
        return rel_name.split(".")[-1].replace("_", " ")
    else:
        rel_true_name = get_relation_true_name(rel_name, kb="depedia")
        tokens = [x for x in rel_true_name.split(" ") if x not in ["ontology", "property"]]
        return " ".join(tokens)

def get_relation_true_name(rel_name, kb="freebase"):
    if kb == "freebase":
        return rel_name.replace("ns:", "").replace(".", " ").replace("_", " ")
    else:
        rel_name = rel_name.strip("<").strip(">")
        if rel_name == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            return "rdf: type"
        prefix = "http://dbpedia.org/"
        tmp = rel_name[len(prefix):].split("/")
        rel_type, rel_true_name = tmp[0], tmp[1]
        tokens = []
        last = 0
        for i, c in enumerate(rel_true_name):
            if c.isupper():
                tokens.append(rel_true_name[last: i])
                last = i
        tokens.append(rel_true_name[last: len(rel_true_name)])
        tokens = [rel_type] + [x for x in tokens if x != ""]
        return " ".join(tokens)

def get_aggregation_true_name(agg_name):
    for i, _agg in enumerate(AGG_NAME):
        if agg_name == _agg:
            return AGG_TRUE_NAME[i]

def get_comparison_true_name(cmp_name):
    for i, _cmp in enumerate(CMP_NAME):
        if cmp_name == _cmp:
            return CMP_TRUE_NAME[i]

def get_order_true_name(ord_name):
    for i, _ord in enumerate(ORD_NAME):
        if ord_name == _ord:
            return ORD_TRUE_NAME[i]

def build_instance_pool_with_gold(instance_pool, gold_aqg, kb, kb_endpoint):
    instance_pool_with_gold = copy.deepcopy(instance_pool)

    # add gold vertex instance to vertex instance set
    for v, v_ins in gold_aqg.v_instances.items():
        _, v_name = v_ins
        v_class = gold_aqg.get_vertex_label(v)
        if v_class == V_CLASS_IDS["var"]:
            continue

        if v_class == V_CLASS_IDS["ent"]:
            v_true_name = get_entity_true_name(v_name, kb, kb_endpoint)
        elif v_class == V_CLASS_IDS["type"]:
            v_true_name = get_type_true_name(v_name)
        elif v_class == V_CLASS_IDS["val"]:
            v_true_name = get_value_true_name(v_name)
        else:
            v_true_name = v_name

        if v_class not in instance_pool_with_gold["vertex"]:
            instance_pool_with_gold["vertex"][v_class] = []

        tmp_pool = [[_v_name, _] for _v_name, _ in instance_pool_with_gold["vertex"][v_class] if _v_name != v_name]
        instance_pool_with_gold["vertex"][v_class] = [[v_name, v_true_name]] + tmp_pool

    # add gold edge instance to edge instance set
    for e, e_ins in gold_aqg.e_instances.items():
        _, e_name = e_ins
        e_class = gold_aqg.get_edge_label(e)

        if e_class % 2 == 1:
            continue

        if e_class not in instance_pool_with_gold["edge"]:
            instance_pool_with_gold["edge"][e_class] = []

        if e_class == E_CLASS_IDS["rel+"]:
            e_true_name = get_relation_true_name(e_name, kb)
        elif e_class == E_CLASS_IDS["agg+"]:
            e_true_name = get_aggregation_true_name(e_name)
        elif e_class == E_CLASS_IDS["cmp+"]:
            e_true_name = get_comparison_true_name(e_name)
        else:
            e_true_name = get_order_true_name(e_name)

        tmp_pool = [[_e_name, _] for _e_name, _ in instance_pool_with_gold["edge"][e_class] if _e_name != e_name]
        instance_pool_with_gold["edge"][e_class] = [[e_name, e_true_name]] + tmp_pool

    return instance_pool_with_gold

def build_instance_pool(data, ent_pool, rel_pool, type_pool, val_pool, kb, kb_endpoint):
    vertex_pool = build_vertex_instance_pool(data, ent_pool, type_pool, val_pool, kb, kb_endpoint)
    edge_pool = build_edge_instance_pool(data, rel_pool, kb)

    instance_pool = {
        "vertex": vertex_pool,
        "edge": edge_pool
    }
    return instance_pool

def build_vertex_instance_pool(data, ent_pool, type_pool, val_pool, kb, kb_endpoint):
    ent_instances = get_entity_instances(data, ent_pool, kb, kb_endpoint)
    type_instances = get_type_instances(data, type_pool, kb)
    val_instances = get_value_instances(data, val_pool)

    vertex_pool = {}

    if ent_instances:
        vertex_pool[V_CLASS_IDS["ent"]] = ent_instances

    if type_instances:
        vertex_pool[V_CLASS_IDS["type"]] = type_instances

    if val_instances:
        vertex_pool[V_CLASS_IDS["val"]] = val_instances

    return  vertex_pool

def build_edge_instance_pool(data, rel_pool, kb):
    rel_instances = get_relation_instances(data, rel_pool, kb)
    agg_instances = get_aggregation_instances()
    cmp_instances = get_comparison_instances()
    ord_instances = get_order_instances()

    edge_pool = {}

    if rel_instances:
        edge_pool[E_CLASS_IDS["rel+"]] = rel_instances

    if agg_instances:
        edge_pool[E_CLASS_IDS["agg+"]] = agg_instances

    if cmp_instances:
        edge_pool[E_CLASS_IDS["cmp+"]] = cmp_instances

    if ord_instances:
        edge_pool[E_CLASS_IDS["ord+"]] = ord_instances

    return edge_pool

def get_entity_instances(data, ent_pool, kb, kb_endpoint):
    """
    use gold entity
    """
    ent_instances = []

    if ent_pool is None:
        if not data["query"]["where"]["union"]:
            data["query"]["where"]["union"].append([])
        for union_conds in data["query"]["where"]["union"]:
            conds = union_conds + data["query"]["where"]["notUnion"]
            for type, cond in conds:
                if type == "Triple":
                    s, p, o = cond
                    if is_entity(s, kb=kb):
                        s_name = s.split("##")[0]
                        s_true_name = get_entity_true_name(s_name, kb, kb_endpoint)
                        ent_instances.append([s_name, s_true_name])
                    if is_entity(o, kb=kb):
                        o_name = o.split("##")[0]
                        o_true_name = get_entity_true_name(o_name, kb, kb_endpoint)
                        ent_instances.append([o_name, o_true_name])
    else:
        if data["id"] in ent_pool:
            for ent_name, ent_true_name in ent_pool[data["id"]]:
                ent_instances.append([ent_name, ent_true_name])
    return ent_instances

def get_type_instances(data, type_pool, kb):
    """
    use gold type
    """
    type_instances = []

    if type_pool is None:
        if not data["query"]["where"]["union"]:
            data["query"]["where"]["union"].append([])
        for union_conds in data["query"]["where"]["union"]:
            conds = union_conds + data["query"]["where"]["notUnion"]
            for _t, cond in conds:
                if _t == "Triple":
                    s, p, o = cond
                    if is_type(o, kb=kb):
                        o_name = o.split("##")[0]
                        o_true_name = get_type_true_name(o_name, kb=kb)
                        type_instances.append([o_name, o_true_name])
    else:
        if data["id"] in type_pool:
            for type_name in type_pool[data["id"]]:
                type_name = "<" + type_name + ">"
                type_true_name = get_type_true_name(type_name, kb=kb)
                type_instances.append([type_name, type_true_name])
    return type_instances

def get_value_instances(data, val_pool=None):

    q = data["question"]
    q = q.strip("?").strip(".").strip(",").strip(" ")

    months = [
        "Jan", "January",
        "Feb", "February",
        "Mar", "March",
        "Apr", "April",
        "May", "May",
        "Jun", "June",
        "Jul", "July",
        "Aug", "August",
        "Sept", "September",
        "Oct", "October",
        "Nov", "November",
        "Dec", "December"
    ]
    month_mapping = {}
    for i, m_name in enumerate(months):
        m = str(i % 2)
        month_mapping[m_name] = '0' + m if len(m) == 1 else m

    superlative_evidences = ["earliest", "last", "latest", "largest", "most", "biggest",
                             "smallest", "fewest", "first", "shortest", "highest", "least",
                             "youngest", "longest"]

    def check_overlap(st, ed):
        for (_st, _ed), _ in val_instances:
            if max(_st, st) <= min(_ed, ed):
                return False
        return True

    def check_word_or_phrase(st, ed):
        if st > 0 and q[st - 1] not in [" "]:
            return False
        if ed < len(q) and q[ed] not in ["?", ",", " "]:
            return False
        return True

    def formalize_date(date):
        _date = date.split("-")
        if len(_date) == 3:
            y, m, d = _date
            if len(y) == 2: y = "20" + y
            if len(m) == 1: m = "0" + m
            if len(d) == 1: d = "0" + d
            res = "\"" + "-".join([y, m, d]) + "\"" + "^^xsd:dateTime"
        elif len(_date) == 2:
            y, m = _date
            if len(y) == 2: y = "20" + y
            if len(m) == 1: m = "0" + m
            res = "\"" + "-".join([y, m]) + "\"" + "^^xsd:dateTime"
        else:
            y = _date[0]
            if len(y) == 2: y = "20" + y
            res = "\"" + "-".join([y]) + "\"" + "^^xsd:dateTime"
        return res

    val_instances = []  # each element is [(st, ed), value], st and ed are the start and end index

    if val_pool:
        for val in val_pool:
            _val = val.strip("\"").replace("\"@en", "").strip("<").strip(">").lower()
            st = q.lower().find(_val)
            if st != -1:
                ed = st + len(_val)
                if check_word_or_phrase(st, ed) and check_overlap(st, ed):
                    val_instances.append([(st, ed), val])

    # 03-02-13 or 03-02-2013 or 03/02/13 or 03/02/2013
    res = re.search(date_pattern_0, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            m, d, y = res.group(0).replace("/", "-").split("-")
            date = formalize_date("-".join([y, m, d]))
            val_instances.append([res.span(0), date])

    # 2013-03-02
    res = re.search(date_pattern_1, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            y, m, d = res.group(0).replace("/", "-").split("-")
            date = formalize_date("-".join([y, m, d]))
            val_instances.append([res.span(0), date])

    # November 6, 1962
    res = re.search(date_pattern_2, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            _date = [x for x in res.group(0).replace(",", " ").split(" ") if x != ""]
            if len(_date) == 3:
                m, d, y = _date
                date = formalize_date("-".join([y, month_mapping[m], d]))
            else:
                m, y = _date
                date = formalize_date("-".join([y, month_mapping[m]]))
            val_instances.append([res.span(0), date])

    # 6 November 1962
    res = re.search(date_pattern_3, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            d, m, y = [x for x in res.group(0).replace(",", " ").split(" ") if x != ""]
            date = formalize_date("-".join([y, month_mapping[m], d]))
            val_instances.append([res.span(0), date])

    # 1995
    res = re.search(year_pattern, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            year = formalize_date(res.group(0))
            date_from = formalize_date(res.group(0) + "-01-01")
            date_to = formalize_date(res.group(0) + "-12-31")
            val_instances.append([res.span(0), year])
            val_instances.append([res.span(0), "$$$".join([date_to, date_from])])

    # 34,245,023
    res = re.search(int_pattern, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            val = res.group(0).replace(",", "")
            val_instances.append([res.span(0), "\"" + val + "\""])

    # 0.05 or -1023.56
    res = re.search(float_pattern, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            val_instances.append([res.span(0), "\"" + res.group(0) + "\""])

    res = re.search(quotation_pattern, q)
    if res:
        st, ed = res.span(0)
        if check_word_or_phrase(st, ed) and check_overlap(st, ed):
            val_instances.append([res.span(0), res.group(0) + "@en"])

    _val_instances = set([val for _, val in val_instances])

    # superlative words
    for word in superlative_evidences:
        if word in q.lower():
            _val_instances.add("\"1\"")

    _val_instances = list(_val_instances)
    _val_instances.sort()
    val_instances = [[val, get_value_true_name(val)] for val in _val_instances]
    return val_instances

def get_relation_instances(data, rel_pool, kb):
    rel_instances = []
    for rel_name in rel_pool[data["id"]]:
        rel_true_name = get_relation_true_name(rel_name, kb=kb)
        rel_instances.append([rel_name, rel_true_name])
    return rel_instances

def get_aggregation_instances():
    agg_instances = []
    for i, agg_name in enumerate(AGG_NAME):
        agg_true_name = AGG_TRUE_NAME[i]
        agg_instances.append([agg_name, agg_true_name])
    return agg_instances

def get_comparison_instances():
    cmp_instances = []
    for i, cmp_name in enumerate(CMP_NAME):
        cmp_true_name = CMP_TRUE_NAME[i]
        cmp_instances.append([cmp_name, cmp_true_name])
    return cmp_instances

def get_order_instances():
    ord_instances = []
    for i, ord_name in enumerate(ORD_NAME):
        ord_true_name = ORD_TRUE_NAME[i]
        ord_instances.append([ord_name, ord_true_name])
    return ord_instances
