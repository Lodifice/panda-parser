from __future__ import print_function
from discodop.containers import Grammar
from discodop.plcfrs import parse
import discodop.pcfg as pcfg
from discodop.kbest import lazykbest
from discodop.estimates import getestimates
from discodop.coarsetofine import prunechart
from parser.parser_interface import AbstractParser
from grammar.lcfrs_derivation import LCFRSDerivation, LCFRSDerivationWrapper
import nltk
from math import log, exp
from parser.trace_manager.trace_manager import add, prod
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.coarse_to_fine_parser.trace_weight_projection import py_edge_weight_projection
from parser.discodop_parser.grammar_adapter import rule_idx_from_label, transform_grammar, transform_grammar_cfg_approx
from parser.trace_manager.sm_trainer import PyLatentAnnotation
import re
from sys import stderr


class DiscodopDerivation(LCFRSDerivation):
    def __init__(self, nltk_tree, grammar):
        """
        :param nltk_tree:
        :type nltk_tree: nltk.Tree
        :param grammar:
        :type grammar: LCFRS
        """
        self.node_counter = 0
        self.rules = {}
        self.children = {}
        self.parent = {}
        self.__init__rec(nltk_tree[0], grammar)
        self.spans = None

    def __init__rec(self, nltk_tree, grammar):
        if isinstance(nltk_tree, str):
            return []

        rule_idx = rule_idx_from_label(nltk_tree.label())

        node = self.node_counter
        self.node_counter += 1
        self.rules[node] = grammar.rule_index(rule_idx)
        self.children[node] = []

        for c in nltk_tree:
            cids = self.__init__rec(c[0], grammar)
            if cids:
                self.children[node] += cids
                self.parent[cids[0]] = node

        return [node]

    def root_id(self):
        return 0

    def getRule(self, id):
        return self.rules[id]

    def child_ids(self, id):
        return self.children[id]

    def child_id(self, id, i):
        return self.children[id][i]

    def position_relative_to_parent(self, id):
        p = self.parent[id]
        return p, self.children[p].index(id)

    def ids(self):
        return range(0, self.node_counter)


class DiscodopKbestParser(AbstractParser):
    def __init__(self,
                 grammar,
                 input=None,
                 save_preprocessing=None,
                 load_preprocessing=None,
                 k=50,
                 heuristics=None,
                 la=None,
                 variational=False,
                 sum_op=False,
                 nontMap=None,
                 cfg_ctf=False,
                 beam_beta=0.0,
                 beam_delta=50,
                 pruning_k=10000,
                 grammarInfo=None,
                 projection_mode=False,
                 latent_viterbi_mode=False,
                 secondaries=None
                 ):
        rule_list = list(transform_grammar(grammar))
        self.disco_grammar = Grammar(rule_list, start=grammar.start())
        self.chart = None
        self.input = input
        self.grammar = grammar
        self.k = k
        self.beam_beta = beam_beta # beam pruning factor, between 0.0 and 1.0; 0.0 to disable.
        self.beam_delta = beam_delta  # maximum span length to which beam_beta is applied
        self.counter = 0
        self.la = la
        self.nontMap = nontMap
        self.variational = variational
        self.op = add if sum_op else prod
        self.debug = False
        self.log_mode = True
        self.estimates = None
        self.cfg_approx = cfg_ctf
        self.pruning_k = pruning_k
        self.grammarInfo = grammarInfo
        self.projection_mode = projection_mode
        self.latent_viterbi_mode = latent_viterbi_mode
        self.secondaries = [] if secondaries is None else secondaries
        self.secondary_mode = "DEFAULT"
        self.k_best_reranker = None
        if grammarInfo is not None:
            if isinstance(self.la, PyLatentAnnotation):
                assert self.la.check_rule_split_alignment()
            else:
                for l in self.la:
                    assert l.check_rule_split_alignment()
        if cfg_ctf:
            cfg_rule_list = list(transform_grammar_cfg_approx(grammar))
            self.disco_cfg_grammar = Grammar(cfg_rule_list, start=grammar.start())
            self.disco_grammar.getmapping(self.disco_cfg_grammar, re.compile('\*[0-9]+$'), None, True, True)
        # self.estimates = 'SXlrgaps', getestimates(self.disco_grammar, 40, grammar.start())

    def best(self):
        pass

    def recognized(self):
        if self.chart and self.chart.root() != 0:
            return True
        else:
            return False

    def max_rule_product_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self.la, variational=False, op=prod)

    def max_rule_sum_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self.la, variational=False,
                                                           op=add)

    def variational_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self, variational=True, op=prod)

    def __projection_based_derivation_tree(self, la, variational=False, op=prod):
        if self.nontMap is None:
            print("A nonterminal map is required for weight projection based parsing!")
            return None
        manager = PyDerivationManager(self.grammar, self.nontMap)
        manager.convert_chart_to_hypergraph(self.chart, self.disco_grammar, debug=False)
        if self.grammarInfo is not None:
            assert manager.is_consistent_with_grammar(self.grammarInfo)
        manager.set_io_cycle_limit(200)
        manager.set_io_precision(0.000001)

        if not isinstance(la, list):
            la = [la]

        edge_weights = None

        for l in la:
            edge_weights_l = py_edge_weight_projection(l, manager, variational=variational, debug=self.debug,
                                                 log_mode=self.log_mode)
            if edge_weights is None:
                edge_weights = edge_weights_l
            else:
                if self.log_mode:
                    edge_weights = [w1 + w2 for w1, w2 in zip(edge_weights, edge_weights_l)]
                else:
                    edge_weights = [op(w1, w2) for w1, w2 in zip(edge_weights, edge_weights_l)]

        if self.debug:
            nans = 0
            infs = 0
            zeros = 0
            for weight in edge_weights:
                if weight == float("nan"):
                    nans += 1
                if weight == float("inf") or weight == float("-inf"):
                    infs += 1
                if weight == 0.0:
                    zeros += 1
            print("[", len(edge_weights), nans, infs, zeros, "]")
            if len(edge_weights) < 100:
                print(edge_weights)

        der = manager.viterbi_derivation(0, edge_weights, self.grammar, op=op, log_mode=self.log_mode)
        if der is None:
            print("p", end="")
            der = self.latent_viterbi_derivation(debug=self.debug)
        if der is not None:
            der = LCFRSDerivationWrapper(der)
        if der is None:
            _, der = next(self.k_best_derivation_trees())
        return der

    def set_secondary_mode(self, mode):
        self.secondary_mode = mode

    def latent_viterbi_derivation(self, debug=False):
        manager = PyDerivationManager(self.grammar, self.nontMap)
        manager.convert_chart_to_hypergraph(self.chart, self.disco_grammar, debug=False)
        if debug:
            manager.serialize(b'/tmp/my_debug_hypergraph.hg')
        if isinstance(self.la, list):
            la = self.la[0]
        else:
            la = self.la
        vit_der = manager.latent_viterbi_derivation(0, la, self.grammar, debug=debug)
        # if len(self.input) < 15 and not debug:
        #     for weight, der in self.k_best_derivation_trees():
        #         if der != vit_der:
        #             print(weight, der, vit_der)
        #             vit_der2 = self.latent_viterbi_derivation(debug=True)
        #             print("vit2", vit_der2)
        #             if vit_der2 != vit_der:
        #                 print("first and second viterbi derivation differ")
        #             if vit_der2 == der:
        #                 print("second viterbi derivation = 1-best-disco-dop derivation")
        #         print("##############################", flush=True)
        #         break
        #         # raise Exception("too much to read")
        if vit_der is not None:
            vit_der = LCFRSDerivationWrapper(vit_der)
        return vit_der

    def best_derivation_tree(self):
        if (self.projection_mode and self.secondary_mode == "DEFAULT") \
                or self.secondary_mode in {"VARIATIONAL", "MAX-RULE-PRODUCT"}:
            variational = self.secondary_mode == "VARIATIONAL" or self.variational and self.secondary_mode == "DEFAULT"
            return self.__projection_based_derivation_tree(self.la, variational=variational, op=self.op)
        elif self.latent_viterbi_mode and self.secondary_mode == "DEFAULT" \
                or self.secondary_mode == "LATENT-VITERBI":
            return self.latent_viterbi_derivation()
        elif self.secondary_mode == "LATENT-RERANK":
            return self.k_best_reranker.best_derivation_tree()
        else:
            for weight, tree in self.k_best_derivation_trees():
                return tree

    def all_derivation_trees(self):
        pass

    def set_input(self, parser_input):
        self.input = parser_input

    def parse(self):
        self.counter += 1
        if self.cfg_approx:
            chart, msg = pcfg.parse(self.input,
                                    self.disco_cfg_grammar,
                                    beam_beta=self.beam_beta,
                                    beam_delta=self.beam_delta)
            if chart:
                chart.filter()
                whitelist, msg = prunechart(chart,
                                            self.disco_grammar,
                                            k=self.pruning_k,
                                            splitprune=True,
                                            markorigin=True,
                                            finecfg=False)
                try:
                    self.chart, msg = parse(self.input,
                                            self.disco_grammar,
                                            estimates=self.estimates,
                                            whitelist=whitelist,
                                            splitprune=True,
                                            markorigin=True,
                                            exhaustive=True)
                except ValueError as e:
                    self.chart = None
                    print("discodop error", e, file=stderr)
        else:
            self.chart, msg = parse(self.input,
                                    self.disco_grammar,
                                    estimates=self.estimates,
                                    beam_beta=self.beam_beta,
                                    beam_delta=self.beam_delta,
                                    exhaustive=True)
        # if self.counter > 86:
        #     print(self.input)
        #     print(self.chart)
        #     print(msg)
        if self.chart:
            self.chart.filter()

    def clear(self):
        self.input = None
        self.chart = None
        if self.k_best_reranker:
            self.k_best_reranker.k_best_list = None
            self.k_best_reranker.ranking = None
            self.k_best_reranker.ranker = None

    def k_best_derivation_trees(self):
        for tree_string, weight in lazykbest(self.chart, self.k):
            try:
                tree = nltk.Tree.fromstring(tree_string)
                yield weight, DiscodopDerivation(tree, self.grammar)
            except ValueError:
                print("\nill-bracketed string:", tree_string, file=stderr)


__all__ = ["DiscodopKbestParser"]
