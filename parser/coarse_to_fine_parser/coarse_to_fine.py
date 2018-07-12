from parser.parser_interface import AbstractParser
from parser.coarse_to_fine_parser.ranker import rank_derivations, build_ranker
from parser.trace_manager.sm_trainer import PyLatentAnnotation
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.coarse_to_fine_parser.trace_weight_projection import py_edge_weight_projection
from parser.trace_manager.trace_manager import add, prod
import math


class Coarse_to_fine_parser(AbstractParser):
    def best_derivation_tree(self):
        if self.mode == "k-best":
            for _, der in self.k_best_derivation_trees():
                return der
        else:
            return self.__projection_based_derivation_tree(self.la[0], variational=self.variational, op=self.op)

    def parse(self):
        self.base_parser.parse()

    def best(self):
        for weight, _ in self.k_best_derivation_trees():
            return weight

    def set_input(self, input):
        self.base_parser.set_input(input)

    def clear(self):
        self.base_parser.clear()
        self.k_best_list = None
        self.ranking = None
        self.ranker = None

    def all_derivation_trees(self):
        pass

    def max_rule_product_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self.la[-1], variational=False, op=prod)

    def max_rule_sum_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self.la[-1], variational=False,
                                                           op=add)

    def variational_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self.la[-1], variational=True, op=prod)

    def __projection_based_derivation_tree(self, la, variational=False, op=prod):
        manager = PyDerivationManager(self.grammar, self.nontMap)
        derivations = [der for _, der in self.base_parser.k_best_derivation_trees()]
        manager.convert_derivations_to_hypergraph(derivations)
        manager.set_io_cycle_limit(200)
        manager.set_io_precision(0.000001)
        self.debug = False
        self.log_mode = True
        edge_weights = py_edge_weight_projection(la, manager, variational=variational, debug=self.debug,
                                                 log_mode=self.log_mode)
        der = manager.viterbi_derivation(0, edge_weights, self.grammar, op=op, log_mode=self.log_mode)
        if der is None:
            if True or self.debug:
                nans = 0
                infs = 0
                zeros = 0
                for weight in edge_weights:
                    if math.isnan(weight):
                        nans += 1
                    if math.isinf(weight):
                        infs += 1
                    if weight == 0.0:
                        zeros += 1
                print("[", len(edge_weights), nans, infs, zeros, "]")
                if len(edge_weights) < 200:
                    print("orig:", edge_weights)
                    edge_weights = py_edge_weight_projection(la, manager, variational=variational, debug=True, log_mode=self.log_mode)
                    print("1:", edge_weights)
                    edge_weights = py_edge_weight_projection(la, manager, variational=variational, debug=True,
                                                             log_mode=self.log_mode)
                    print("2:", edge_weights)

            print("p", end="")
            _, der = next(self.k_best_derivation_trees())
        return der

    def recognized(self):
        return self.base_parser.recognized()

    def __init__(self,
                 grammar,
                 la,
                 grammarInfo,
                 nontMap,
                 base_parser_type=None,
                 base_parser=None,
                 input=None,
                 save_preprocessing=None,
                 load_preprocessing=None,
                 k=50,
                 heuristics=-1.0,
                 mode="k-best",
                 sum_op=False,
                 variational=False):
        self.grammar = grammar
        if base_parser is not None:
            self.base_parser = base_parser
        elif base_parser_type is not None:
            self.base_parser = base_parser_type(grammar, input=input, save_preprocessing=save_preprocessing, load_preprocessing=load_preprocessing, k=k, heuristics=heuristics)
        else:
            assert False and "Neither base_parser_type nor base_parser specified."

        self.la = [la] if isinstance(la, PyLatentAnnotation) else la
        self.grammarInfo = grammarInfo
        self.nontMap = nontMap
        self.ranking = None
        self.ranker = None
        self.k_best_list = None
        self.mode = mode
        self.op = add if sum_op else prod
        self.variational = variational

    def k_best_derivation_trees(self):
        if self.k_best_list is None:

            self.k_best_list = []
            self.ranking = []
            for i, (weight, der) in enumerate(self.base_parser.k_best_derivation_trees()):
                self.k_best_list.append(der)
                self.ranking.append((i, weight))

            length = len(self.ranking)
            for ref, la in enumerate(self.la):
                if self.ranker is None:
                    self.ranker = build_ranker(self.k_best_list, self.grammar, self.grammarInfo, self.nontMap)

                new_ranking = self.ranker.rank(la)
                self.ranker.clean_up()

                # a simple heuristic, one could also back-off between those lists
                i = 0
                if new_ranking[-1][1] == 0.0:
                    # todo: some binary search could be used here for efficency
                    while i < len(new_ranking) and new_ranking[i][1] > 0.0:
                        i += 1
                    if i < length * 0.1:
                        break
                length = i
                self.ranking = new_ranking
                # print "Ranking: ", ref, self.ranking

            # print

        for idx, weight in self.ranking:
            # if not (math.isinf(weight) or math.isnan(weight)):
            yield weight, self.k_best_list[idx]

    def resolve_path(self, preprocess_path):
        return self.base_parser.resolve_path(preprocess_path)


__all__ = ["Coarse_to_fine_parser"]
