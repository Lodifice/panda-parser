from __future__ import print_function

import copy
import unittest
from sys import stderr

from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from dependency.induction import induce_grammar
from dependency.labeling import the_labeling_factory
from dependency.minimum_risk import compute_minimum_risk_tree
from dependency.oracle import compute_oracle_tree
from grammar.induction.recursive_partitioning import direct_extraction, cfg
from grammar.induction.terminal_labeling import the_terminal_labeling_factory
from grammar.lcfrs_derivation import derivation_to_hybrid_tree
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
from parser.coarse_to_fine_parser.coarse_to_fine import Coarse_to_fine_parser
from parser.gf_parser.gf_export import *
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from parser.sDCPevaluation.evaluator import DCP_evaluator, dcp_to_hybridtree
from parser.trace_manager.sm_trainer import build_PyLatentAnnotation_initial
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager
from tests.test_induction import hybrid_tree_1, hybrid_tree_2
from util.enumerator import Enumerator


class GrammaticalFrameworkTest(unittest.TestCase):
    def test_grammar_export(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        _, grammar = induce_grammar([tree, tree2],
                                    the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                    # the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel'),
                                    terminal_labeling.token_label, [direct_extraction], 'START')
        print(max([grammar.fanout(nont) for nont in grammar.nonts()]))
        print(grammar)

        prefix = '/tmp/'
        name = 'tmpGrammar'

        name_ = export(grammar, prefix, name)

        self.assertEqual(0, compile_gf_grammar(prefix, name_))

        GFParser.preprocess_grammar(grammar)

        string = ["NP", "N", "V", "V", "V"]

        parser = GFParser(grammar, string)

        self.assertTrue(parser.recognized())

        der = parser.best_derivation_tree()
        self.assertTrue(der.check_integrity_recursive(der.root_id(), grammar.start()))

        print(der)

        print(derivation_to_hybrid_tree(der, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

        dcp = DCP_evaluator(der).getEvaluation()

        h_tree_2 = HybridTree()
        token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                          zip('Piet Marie helpen lezen leren'.split(' '), 'NP N V V V'.split(' '))]
        dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                          construct_conll_token)

        print(h_tree_2)

    def test_k_best_parsing(self):
        limit_train = 20
        limit_test = 10
        train = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        parser_type = GFParser_k_best
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim)
        tree_yield = term_labelling.prepare_parser_input

        trees = parse_conll_corpus(test, False, limit_test)

        for i, tree in enumerate(trees):
            print("Parsing sentence ", i, file=stderr)

            # print >>stderr, tree

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()), k=50)

            self.assertTrue(parser.recognized())

            derivations = [der for der in parser.k_best_derivation_trees()]
            print("# derivations: ", len(derivations), file=stderr)
            h_trees = []
            current_weight = 0
            weights = []
            derivation_list = []
            for weight, der in derivations:
                # print >>stderr, exp(-weight)
                # print >>stderr, der

                self.assertTrue(not der in derivation_list)

                derivation_list.append(der)

                # TODO this should hold, but it looks like a GF bug!
                # self.assertGreaterEqual(weight, current_weight)
                current_weight = weight

                dcp = DCP_evaluator(der).getEvaluation()
                h_tree = HybridTree()
                cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                dcp_to_hybridtree(h_tree, dcp, cleaned_tokens, False, construct_conll_token)

                h_trees.append(h_tree)
                weights.append(weight)

                # print >>stderr, h_tree

            # print a matrix indicating which derivations result
            # in the same hybrid tree
            if True:
                for i, h_tree1 in enumerate(h_trees):
                    for h_tree2 in h_trees:
                        if h_tree1 == h_tree2:
                            print("x", end=' ', file=stderr)
                        else:
                            print("", end=' ', file=stderr)
                    print(weights[i], file=stderr)
                print(file=stderr)

    def test_minimum_risk_parsing(self):
        limit_train = 20
        limit_test = 10
        train = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        parser_type = GFParser_k_best
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim)
        tree_yield = term_labelling.prepare_parser_input

        trees = parse_conll_corpus(test, False, limit_test)

        for i, tree in enumerate(trees):
            print("Parsing sentence ", i, file=stderr)

            # print >>stderr, tree

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()), k=50)

            self.assertTrue(parser.recognized())

            derivations = [der for der in parser.k_best_derivation_trees()]
            print("# derivations: ", len(derivations), file=stderr)
            h_trees = []
            current_weight = 0
            weights = []
            derivation_list = []
            for weight, der in derivations:
                self.assertTrue(not der in derivation_list)

                derivation_list.append(der)

                dcp = DCP_evaluator(der).getEvaluation()
                h_tree = HybridTree()
                cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                dcp_to_hybridtree(h_tree, dcp, cleaned_tokens, False, construct_conll_token)

                h_trees.append(h_tree)
                weights.append(weight)

            if True:
                min_risk_tree = compute_minimum_risk_tree(h_trees, weights)
                if not min_risk_tree.__eq__(h_trees[0]):
                    print(h_trees[0])
                    print(min_risk_tree)

    def test_oracle_parsing(self):
        limit_train = 20
        limit_test = 10
        train = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        parser_type = GFParser_k_best
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim)
        tree_yield = term_labelling.prepare_parser_input

        trees = parse_conll_corpus(test, False, limit_test)

        for i, tree in enumerate(trees):
            print("Parsing sentence ", i, file=stderr)

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()), k=50)

            self.assertTrue(parser.recognized())

            derivations = [der for der in parser.k_best_derivation_trees()]
            print("# derivations: ", len(derivations), file=stderr)
            h_trees = []
            current_weight = 0
            weights = []
            derivation_list = []
            for weight, der in derivations:
                self.assertTrue(not der in derivation_list)

                derivation_list.append(der)

                dcp = DCP_evaluator(der).getEvaluation()
                h_tree = HybridTree()
                cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                dcp_to_hybridtree(h_tree, dcp, cleaned_tokens, False, construct_conll_token)

                h_trees.append(h_tree)
                weights.append(weight)

            if True:
                oracle_tree = compute_oracle_tree(h_trees, tree)
                if not oracle_tree.__eq__(h_trees[0]):
                    print(h_trees[0], file=stderr)
                    print(oracle_tree, file=stderr)

    def test_best_trees(self):
        limit_train = 5000
        limit_test = 100
        train = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        parser_type = GFParser_k_best
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("child", "pos+deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim)
        tree_yield = term_labelling.prepare_parser_input

        trees = parse_conll_corpus(test, False, limit_test)

        for i, tree in enumerate(trees):
            print("Parsing sentence ", i, file=stderr)

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()), k=200)

            self.assertTrue(parser.recognized())

            viterbi_weight = parser.viterbi_weight()
            viterbi_deriv = parser.viterbi_derivation()

            der_to_tree = lambda der: dcp_to_hybridtree(HybridTree(), DCP_evaluator(der).getEvaluation(),
                                                        copy.deepcopy(tree.full_token_yield()), False,
                                                        construct_conll_token)

            viterbi_tree = der_to_tree(viterbi_deriv)

            ordered_parse_trees = parser.best_trees(der_to_tree)

            best_tree, best_weight, best_witnesses = ordered_parse_trees[0]

            for i, (parsed_tree, _, _) in enumerate(ordered_parse_trees):
                if parsed_tree.__eq__(tree):
                    print("Gold tree is ", i + 1, " in best tree list", file=stderr)
                    break

            if (not viterbi_tree.__eq__(best_tree) and viterbi_weight != best_weight):
                print("viterbi and k-best tree differ", file=stderr)
                print("viterbi: ", viterbi_weight, file=stderr)
                print("k-best: ", best_weight, best_witnesses, file=stderr)
                if False:
                    print(viterbi_tree, file=stderr)
                    print(tree_to_conll_str(viterbi_tree), file=stderr)
                    print(best_tree, file=stderr)
                    print(tree_to_conll_str(best_tree), file=stderr)
                    print("gold tree", file=stderr)
                    print(tree, file=stderr)
                    print(tree_to_conll_str(tree), file=stderr)

    def test_projection_based_parser_k_best_hack(self):
        grammar = self.build_grammar()
        inp = ["a"] * 3
        nontMap = Enumerator()
        gi = PyGrammarInfo(grammar, nontMap)
        sm = PyStorageManager()
        la = build_PyLatentAnnotation_initial(grammar, gi, sm)

        parser = Coarse_to_fine_parser(grammar, la, gi, nontMap, base_parser_type=GFParser_k_best)
        parser.set_input(inp)
        parser.parse()
        self.assertTrue(parser.recognized())
        der = parser.max_rule_product_derivation()
        print(der)

        der = parser.best_derivation_tree()
        print(der)

        for node in der.ids():
            print(der.getRule(node), der.spanned_ranges(node))


if __name__ == '__main__':
    unittest.main()
