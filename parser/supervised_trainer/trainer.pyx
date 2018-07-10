#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from cython.operator cimport dereference as deref
from grammar.derivation_interface import AbstractDerivation
from grammar.rtg import RTG
from parser.discodop_parser.grammar_adapter import rule_idx_from_label, striplabelre, unescape_brackets
from discodop.containers import Chart
from typing import Iterable
from discodop.containers import Grammar


cdef extern from "Trainer/TraceManager.h" namespace "Trainer":
    cdef void add_hypergraph_to_trace[Nonterminal, TraceID](
            TraceManagerPtr[Nonterminal, TraceID] manager
            , shared_ptr[Hypergraph[Nonterminal, size_t]] hypergraph
            , Element[Node[Nonterminal]] root
            , double frequency)


cdef class PyElement:
    cdef shared_ptr[Element[Node[NONTERMINAL]]] element
    def __cinit__(self):
        self.element = shared_ptr[Element[Node[NONTERMINAL]]]()
         # self.element = make_shared[Element[Node[NONTERMINAL]]](element)


cdef class PyDerivationManager(PyTraceManager):
    def __init__(self, grammar, Enumerator nonterminal_map=None):
        """
        :param grammar:
        :type grammar: PyLCFRS
        """
        if nonterminal_map is None:
            nonterminal_map = Enumerator()
            for nont in grammar.nonts():
                nonterminal_map.object_index(nont)
        cdef vector[NONTERMINAL] node_labels = range(0, nonterminal_map.counter)
        self.node_labels = make_shared[vector[NONTERMINAL]](node_labels)
        cdef vector[size_t] edge_labels = range(0, len(grammar.rule_index()))
        self.edge_labels = make_shared[vector[size_t]](edge_labels)
        self.nonterminal_map = nonterminal_map

        self.trace_manager = build_trace_manager_ptr[NONTERMINAL, size_t](self.node_labels
            , self.edge_labels
            , False)

    cpdef void convert_derivations_to_hypergraphs(self, corpus, float frequency=1.0):
        """
        :type corpus: Iterable[AbstractDerivation]
        :param frequency: frequency with which each derivation will be accounted by EM training
        :type frequency: float
        Converts each derivation in **corpus** in an equivalent hypergraph and adds it to the trace manager.
        """
        cdef shared_ptr[Hypergraph[NONTERMINAL, size_t]] hg
        cdef vector[Element[Node[NONTERMINAL]]] sources
        cdef PyElement pyElement

        for derivation in corpus:
            assert(isinstance(derivation, AbstractDerivation))
            hg = make_shared[Hypergraph[NONTERMINAL, size_t]](self.node_labels, self.edge_labels)
            nodeMap = {}

            # create nodes
            for node in derivation.ids():
                nont = derivation.getRule(node).lhs().nont()
                nLabel = self.nonterminal_map.object_index(nont)
                pyElement2 = PyElement()
                pyElement2.element = make_shared[Element[Node[NONTERMINAL]]](deref(hg).create(nLabel))
                nodeMap[node] = pyElement2

            # create edges
            for node in derivation.ids():
                eLabel = derivation.getRule(node).get_idx()
                for child in derivation.child_ids(node):
                    # nont = derivation.getRule(nont).lhs().nont()
                    pyElement = nodeMap[child]
                    sources.push_back(deref(pyElement.element))

                # target
                pyElement = nodeMap[node]
                deref(hg).add_hyperedge(eLabel, deref(pyElement.element), sources)
                sources.clear()

            # root
            pyElement = nodeMap[derivation.root_id()]
            add_hypergraph_to_trace[NONTERMINAL, size_t](self.trace_manager, hg, deref(pyElement.element), 1.0)
            # nodeMap.clear()

    def __compute_node_map_key(self, node, derivation):
            nont = derivation.getRule(node).lhs().nont()
            spans = derivation.spanned_ranges(node)
            return nont, tuple(spans)

    cpdef void convert_derivations_to_hypergraph(self, corpus, float frequency=1.0):
        """
        :param corpus: nonempty iterable of derivations for the same sentence   
        :type corpus: Iterable[AbstractDerivation]
        :param frequency: frequency with which the hypergraph will be accounted by EM training
        :type frequency: float
        Joins an iterator over derivations of a single sentence into a single packed hypergraph. 
        The nodes of the hypergraph are nonterminals from the rules annotated by the string positions they span. 
        Duplicate edges which may arise this way are removed.
        """
        cdef shared_ptr[Hypergraph[NONTERMINAL, size_t]] hg
        cdef vector[Element[Node[NONTERMINAL]]] sources
        cdef PyElement pyElement

        hg = make_shared[Hypergraph[NONTERMINAL, size_t]](self.node_labels, self.edge_labels)
        nodeMap = {}
        edgeSet = set()
        root_key = None

        for derivation in corpus:
            assert(isinstance(derivation, AbstractDerivation))
            # create nodes
            for node in derivation.ids():
                nont, spans = self.__compute_node_map_key(node, derivation)
                if (nont, spans) not in nodeMap:
                    nLabel = self.nonterminal_map.object_index(nont)
                    pyElement2 = PyElement()
                    pyElement2.element = make_shared[Element[Node[NONTERMINAL]]](deref(hg).create(nLabel))
                    nodeMap[(nont, spans)] = pyElement2

            if root_key is None:
                root_key = self.__compute_node_map_key(derivation.root_id(), derivation)
            else:
                assert root_key == self.__compute_node_map_key(derivation.root_id(), derivation)

            # create edges
            for node in derivation.ids():
                target_key = self.__compute_node_map_key(node, derivation)
                eLabel = derivation.getRule(node).get_idx()
                source_keys = []
                for child in derivation.child_ids(node):
                    source_keys.append(self.__compute_node_map_key(child, derivation))

                edge_key = target_key, eLabel, tuple(source_keys)
                if edge_key not in edgeSet:
                    edgeSet.add(edge_key)

                    for source_key in source_keys:
                        pyElement = nodeMap[source_key]
                        sources.push_back(deref(pyElement.element))

                    # target
                    pyElement = nodeMap[target_key]
                    deref(hg).add_hyperedge(eLabel, deref(pyElement.element), sources)
                    sources.clear()

        # root
        assert root_key is not None
        pyElement = nodeMap[root_key]
        add_hypergraph_to_trace[NONTERMINAL, size_t](self.trace_manager, hg, deref(pyElement.element), frequency)

    cpdef Enumerator get_nonterminal_map(self):
        return self.nonterminal_map

    cpdef void convert_rtgs_to_hypergraphs(self, rtgs, float frequency=1.0):
        """
        :param rtgs: an iterator over reduct RTG of the grammar passed in the constructor, i.e.,\n 
                     - nonterminals are of the the form (N, X) where N is a nonterminal in grammar,\n 
                     - the initial nonterminal is the start symbol of grammar,\n 
                     - rules are of the from (N, X) -> n((N_1, X_1) … (N_k, X_k)) where 
                       N -> N_1 … N_k is the rule with idx n in grammar
        :type rtgs: Iterable[RTG]
        :param frequency: frequency with which each rtg will be accounted by EM training
        :type frequency: float
        Converts each RTG in an equivalent hypergraph and adds it to the trace manager.
        """
        cdef shared_ptr[Hypergraph[NONTERMINAL, size_t]] hg
        cdef vector[Element[Node[NONTERMINAL]]] sources
        cdef PyElement pyElement

        for rtg in rtgs:
            assert(isinstance(rtg, RTG))
            hg = make_shared[Hypergraph[NONTERMINAL, size_t]](self.node_labels, self.edge_labels)
            nodeMap = {}

            # create nodes
            for nont in rtg.nonterminals:
                orig_nont = nont[0]
                nLabel = self.nonterminal_map.object_index(orig_nont)
                pyElement2 = PyElement()
                pyElement2.element = make_shared[Element[Node[NONTERMINAL]]](deref(hg).create(nLabel))
                nodeMap[nont] = pyElement2

            # create edges
            for rule in rtg.rules:
                eLabel = rule.symbol
                for rhs_nont in rule.rhs:
                    pyElement = nodeMap[rhs_nont]
                    sources.push_back(deref(pyElement.element))

                # target
                pyElement = nodeMap[rule.lhs]
                deref(hg).add_hyperedge(eLabel, deref(pyElement.element), sources)
                sources.clear()

            # root
            pyElement = nodeMap[rtg.initial]
            add_hypergraph_to_trace[NONTERMINAL, size_t](self.trace_manager, hg, deref(pyElement.element), frequency)
            # nodeMap.clear()

    cpdef void convert_chart_to_hypergraph(self, chart, disco_grammar, bint debug=False) except +:
        """
        :param chart: a populated Chart from the discodop parser (obtained with *disco_grammar*) 
        :type chart: Chart
        :param disco_grammar: a discodop grammar
        :type disco_grammar: Grammar
        Converts the chart in an equivalent hypergraph and adds it to the trace manager.
        """
        cdef shared_ptr[Hypergraph[NONTERMINAL, size_t]] hg
        cdef vector[Element[Node[NONTERMINAL]]] sources
        cdef PyElement pyElement
        cdef int node_intermediate
        cdef int node_prim
        cdef int node
        cdef int eLabel
        cdef set intermediate_nodes
        cdef int edge_num, edge_num_prim

        hg = make_shared[Hypergraph[NONTERMINAL, size_t]](self.node_labels, self.edge_labels)
        nodeMap = {}
        intermediate_nodes = set()

        # create nodes
        for node in range(1, chart.numitems() + 1):
            orig_nont = disco_grammar.nonterminalstr(chart.label(node))
            orig_nont = unescape_brackets(orig_nont)
            # print(orig_nont)
            if striplabelre.match(orig_nont):
                intermediate_nodes.add(node)
                continue

            assert orig_nont in self.nonterminal_map.obj_to_ind
            nLabel = self.nonterminal_map.object_index(orig_nont)
            pyElement2 = PyElement()
            pyElement2.element = make_shared[Element[Node[NONTERMINAL]]](deref(hg).create(nLabel))
            nodeMap[node] = pyElement2

        # create edges
        for node_prim in range(1, chart.numitems() + 1):
            # skip intermediate nodes
            if node_prim in intermediate_nodes:
                continue
            # print("node prim", node_prim)
            # go over intermediate unary edges
            for edge_num_prim in range(chart.numedges(node_prim)):
                edge = chart.getEdgeForItem(node_prim, edge_num_prim)
                # print("edge", edge)
                assert isinstance(edge, tuple)
                assert edge[2] == 0


                # determine intermediate node
                node_intermediate = edge[1]

                # create hyperedges for primary node for each edge outgoing from intermediate node
                for edge_num in range(chart.numedges(node_intermediate)):
                    eLabel = rule_idx_from_label(disco_grammar.nonterminalstr(chart.label(node_intermediate)))

                    if debug:
                        print("goal", node_prim, "edge", eLabel, "sources:", end=" ")
                    edge_inter = chart.getEdgeForItem(node_intermediate, edge_num)
                    if isinstance(edge_inter, tuple):
                        for rhs_nont in [j for j in [edge_inter[1], edge_inter[2]] if j != 0]:
                            if debug:
                                print(rhs_nont, end=" ")
                            pyElement = nodeMap[rhs_nont]
                            sources.push_back(deref(pyElement.element))
                    if debug:
                        print()
                    # target
                    pyElement = nodeMap[node_prim]
                    deref(hg).add_hyperedge(eLabel, deref(pyElement.element), sources)
                    sources.clear()


        # root
        pyElement = nodeMap[chart.root()]
        add_hypergraph_to_trace[NONTERMINAL, size_t](self.trace_manager, hg, deref(pyElement.element), 1.0)
        # nodeMap.clear()


__all__ = ["PyDerivationManager"]