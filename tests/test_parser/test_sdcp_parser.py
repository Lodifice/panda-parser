from __future__ import print_function

import unittest
from sys import stderr

from constituent.induction import fringe_extract_lcfrs, direct_extract_lcfrs
from corpora.conll_parse import parse_conll_corpus
from dependency.induction import induce_grammar
from dependency.labeling import the_labeling_factory
from grammar.induction.decomposition import fanout_limited_partitioning
from grammar.induction.recursive_partitioning import cfg
from grammar.induction.terminal_labeling import the_terminal_labeling_factory, PosTerminals, FormPosTerminalsUnk, FormTerminals
from grammar.lcfrs import LCFRS
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token, construct_constituent_token
from parser.sDCP_parser.playground import split_merge_training
from parser.sDCP_parser.sdcp_parser_wrapper import PysDCPParser, LCFRS_sDCP_Parser, SDCPDerivation
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
from parser.sDCPevaluation.evaluator import dcp_to_hybridtree, DCP_evaluator
from parser.trace_manager.sm_trainer import PyEMTrainer
from tests.test_induction import hybrid_tree_1, hybrid_tree_2


class sDCPParserTest(unittest.TestCase):
    def test_basic_sdcp_parsing_dependency(self):
        tree1 = hybrid_tree_1()
        tree2 = hybrid_tree_2()

        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree1, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        print("grammar induced. Printing rules...", file=stderr)

        for rule in grammar.rules():
            print(rule, file=stderr)

        parser_type = LCFRS_sDCP_Parser

        print("preprocessing grammar", file=stderr)

        parser_type.preprocess_grammar(grammar, terminal_labeling)

        print("invoking parser", file=stderr)

        parser = parser_type(grammar, tree1)

        print("listing derivations", file=stderr)

        for der in parser.all_derivation_trees():
            print(der)
            output_tree = HybridTree()
            tokens = tree1.token_yield()
            dcp_to_hybridtree(output_tree, DCP_evaluator(der).getEvaluation(), tokens, False, construct_conll_token)
            print(tree1)
            print(output_tree)

        print("completed test", file=stderr)

    def test_basic_sdcp_parsing_constituency(self):
        tree1 = constituent_tree_1()
        tree2 = constituent_tree_2()
        tree3 = constituent_tree_1_pos_stripped()

        terminal_labeling = FormTerminals() # [tree1, tree2], 1, filter=["VP"])
        fanout = 1

        grammar = LCFRS('START')
        for tree in [tree1, tree2]:
            tree_part = tree.unlabelled_structure()
            part = fanout_limited_partitioning(tree_part, fanout)
            tree_grammar = fringe_extract_lcfrs(tree, part, naming='child', term_labeling=terminal_labeling)
            grammar.add_gram(tree_grammar)
        grammar.make_proper()

        print("grammar induced. Printing rules...", file=stderr)

        for rule in grammar.rules():
            print(rule, file=stderr)

        parser_type = LCFRS_sDCP_Parser

        print("preprocessing grammar", file=stderr)

        parser_type.preprocess_grammar(grammar, terminal_labeling, debug=True)

        print("invoking parser", file=stderr)

        parser = parser_type(grammar, tree1)

        print("listing derivations", file=stderr)

        for der in parser.all_derivation_trees():
            print(der)
            output_tree = ConstituentTree(tree1.sent_label())
            tokens = [construct_constituent_token(token.form(), '--', True) for token in tree1.token_yield()]
            dcp_to_hybridtree(output_tree, DCP_evaluator(der).getEvaluation(), tokens, False,
                              construct_constituent_token)
            print(tree1)
            print(output_tree)

        parser = parser_type(grammar, tree3)
        print(parser.recognized())
        for der in parser.all_derivation_trees():
            print(der)
            output_tree = ConstituentTree(tree3.sent_label())
            tokens = [construct_constituent_token(token.form(), '--', True) for token in tree3.token_yield()]
            dcp_to_hybridtree(output_tree, DCP_evaluator(der).getEvaluation(), tokens, False,
                              construct_constituent_token)
            print(tree3)
            print(output_tree)

        print("completed test", file=stderr)

    def test_basic_em_training(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print(rule, file=stderr)

        print("compute reducts", file=stderr)

        trace = compute_reducts(grammar, [tree, tree2], terminal_labeling)

        print("call em Training", file=stderr)
        emTrainer = PyEMTrainer(trace)
        emTrainer.em_training(grammar, n_epochs=10)

        print("finished em Training", file=stderr)

        for rule in grammar.rules():
            print(rule, file=stderr)

    def test_corpus_em_training(self):
        train = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        limit_train = 200
        test = train
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        # for rule in grammar.rules():
        #    print >>stderr, rule

        trees = parse_conll_corpus(train, False, limit_train)

        print("compute reducts", file=stderr)

        trace = compute_reducts(grammar_prim, trees, term_labelling)

        print("call em Training", file=stderr)
        emTrainer = PyEMTrainer(trace)
        emTrainer.em_training(grammar_prim, 20, tie_breaking=True, init="equal", sigma=0.05, seed=50)

        print("finished em Training", file=stderr)

        # for rule in grammar.rules():
        #     print >>stderr, rule

    def test_corpus_sdcp_parsing(self):
        # parser_type = PysDCPParser
        print("testing (plain) sDCP parser", file=stderr)
        self.generic_parsing_test(PysDCPParser, 50, 20, False)

    def test_corpus_lcfrs_sdcp_parsing(self):
        print("testing LCFRS/sDCP hybrid parser", file=stderr)
        self.generic_parsing_test(LCFRS_sDCP_Parser, 2000, 100, True)

    def generic_parsing_test(self, parser_type, limit_train, limit_test, compare_order):
        def filter_by_id(n, trees):
            j = 0
            for tree in trees:
                if j in n:
                    yield tree
                j += 1

        # params
        train = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        # test = 'res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim, term_labelling)

        trees = parse_conll_corpus(test, False, limit_test)

        count_derivs = {}
        no_complete_match = 0

        for i, tree in enumerate(trees):
            print("Parsing tree for ", i, file=stderr)

            print(tree, file=stderr)

            parser = parser_type(grammar_prim, tree)
            self.assertTrue(parser.recognized())
            count_derivs[i] = 0

            print("Found derivations for ", i, file=stderr)
            j = 0

            derivations = []

            for der in parser.all_derivation_trees():
                self.assertTrue(der.check_integrity_recursive(der.root_id(), start))

                print(count_derivs[i], file=stderr)
                print(der, file=stderr)

                output_tree = HybridTree()
                tokens = tree.token_yield()

                the_yield = der.compute_yield()
                # print >>stderr, the_yield
                tokens2 = list(map(lambda pos: construct_conll_token('_', pos), the_yield))

                dcp_to_hybridtree(output_tree, DCP_evaluator(der).getEvaluation(), tokens2, False,
                                  construct_conll_token, reorder=False)
                print(tree, file=stderr)
                print(output_tree, file=stderr)

                self.compare_hybrid_trees(tree, output_tree, compare_order)
                count_derivs[i] += 1
                derivations.append(der)

            self.assertTrue(sDCPParserTest.pairwise_different(derivations, sDCPParserTest.compare_derivations))
            self.assertEqual(len(derivations), count_derivs[i])

            if count_derivs[i] == 0:
                no_complete_match += 1

        for key in count_derivs:
            print(key, count_derivs[key])

        print("# trees with no complete match:", no_complete_match)

    def compare_hybrid_trees(self, tree1, tree2, compare_order=False):
        self.assertTrue(isinstance(tree1, HybridTree))
        self.assertTrue(isinstance(tree2, HybridTree))
        self.compare_hybrid_trees_rec(tree1, tree1.root, tree2, tree2.root, compare_order)

    def compare_hybrid_trees_rec(self, tree1, ids1, tree2, ids2, compare_order):
        self.assertEqual(len(ids1), len(ids2))
        for id1, id2 in zip(ids1, ids2):
            token1 = tree1.node_token(id1)
            token2 = tree2.node_token(id2)
            self.assertEqual(token1.pos(), token2.pos())
            self.assertEqual(token1.deprel(), token2.deprel())
            if (compare_order):
                self.assertEqual(tree1.in_ordering(id1), tree2.in_ordering(id2))
                if (tree1.in_ordering(id1)):
                    self.assertEqual(tree1.node_index(id1), tree2.node_index(id2))
            self.compare_hybrid_trees_rec(tree1, tree1.children(id1), tree2, tree2.children(id2), compare_order)

    @staticmethod
    def compare_derivations(der1, der2):
        assert isinstance(der1, SDCPDerivation)
        assert isinstance(der2, SDCPDerivation)
        # return str(der1) == str(der2)

        id1 = der1.root_id()
        id2 = der2.root_id()

        sDCPParserTest.compare_derivation_recursive(der1, id1, der2, id2)

    @staticmethod
    def compare_derivation_recursive(der1, id1, der2, id2):
        """
        :param der1:
        :type der1: SDCPDerivation
        :param id1:
        :type id1:
        :param der2:
        :type der2: SDCPDerivation
        :param id2:
        :type id2:
        :return:
        :rtype:
        """
        if not der1.getRule(id1) == der2.getRule(id2):
            return False
        if not len(der1.child_ids(id1)) == len(der2.child_ids(id2)):
            return False
        for i in range(len(der1.child_ids(id1))):
            if not sDCPParserTest.compare_derivation_recursive(der1, der1.child_id(id1, i), der2,
                                                               der2.child_id(id2, i)):
                return False
        return True

    @staticmethod
    def pairwise_different(l, comparator):
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                if comparator(l[i], l[j]):
                    return False
        return True

    def test_trace_serialization(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print(rule, file=stderr)

        trace = compute_reducts(grammar, [tree, tree2], terminal_labeling)
        trace.serialize(b"/tmp/reducts.p")

        grammar_load = grammar
        trace2 = PySDCPTraceManager(grammar_load, terminal_labeling)
        trace2.load_traces_from_file(b"/tmp/reducts.p")
        trace2.serialize(b"/tmp/reducts2.p")

        with open(b"/tmp/reducts.p", "r") as f1, open(b"/tmp/reducts2.p", "r") as f2:
            for e1, e2 in zip(f1, f2):
                self.assertEqual(e1, e2)

    def test_basic_split_merge(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print(rule, file=stderr)

        print("call S/M Training", file=stderr)

        new_grammars = split_merge_training(grammar, terminal_labeling, [tree, tree2], 3, 5, merge_threshold=0.5,
                                            debug=False)

        for new_grammar in new_grammars:
            for i, rule in enumerate(new_grammar.rules()):
                print(i, rule, file=stderr)
            print(file=stderr)

        print("finished S/M Training", file=stderr)

    def test_corpus_split_merge_training(self):
        train = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        limit_train = 100
        test = train
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        # for rule in grammar.rules():
        #    print >>stderr, rule

        trees = parse_conll_corpus(train, False, limit_train)
        print("call S/M Training", file=stderr)

        new_grammars = split_merge_training(grammar_prim, term_labelling, trees, 4, 10, tie_breaking=True, init="equal",
                                            sigma=0.05, seed=50, merge_threshold=0.1)

        print("finished S/M Training", file=stderr)

        for new_grammar in new_grammars:
            for i, rule in enumerate(new_grammar.rules()):
                print(i, rule, file=stderr)
            print(file=stderr)

    def test_lcfrs_sdcp_parsing(self):
        def tree1():
            tree = ConstituentTree("1")
            for i, t in enumerate(["a", "b", "c", "d"]):
                tree.add_leaf(str(i), "P" + t, t)
            tree.set_label('r0', 'C')
            tree.set_label('r1', 'A')
            tree.set_label('r2', 'B')
            tree.add_to_root('r0')
            tree.add_child('r0', 'r1')
            tree.add_child('r0', 'r2')
            tree.add_child('r1', '0')
            tree.add_child('r1', '2')
            tree.add_child('r2', '1')
            tree.add_child('r2', '3')
            print(tree, tree.word_yield())
            return tree

        def tree2():
            tree = ConstituentTree("1")
            for i, t in enumerate(["a", "b", "d", "c"]):
                tree.add_leaf(str(i), "P" + t, t)
            tree.set_label('r0', 'C')
            tree.set_label('r1', 'A')
            tree.set_label('r2', 'B')
            tree.add_to_root('r0')
            tree.add_child('r0', 'r1')
            tree.add_child('r0', 'r2')
            tree.add_child('r1', '0')
            tree.add_child('r1', '3')
            tree.add_child('r2', '1')
            tree.add_child('r2', '2')
            print(tree, tree.word_yield())
            return tree

        t1 = tree1()
        t2 = tree2()

        grammar = direct_extract_lcfrs(t1)
        grammar.add_gram(direct_extract_lcfrs(t2))

        print(grammar)
        # LCFRS_sDCP_Parser.preprocess_grammar(grammar, PosTerminals(), debug=True)

        parser = LCFRS_sDCP_Parser(grammar, terminal_labelling=PosTerminals(), debug=True)
        for t in [t1, t2]:
            # parser = LCFRS_sDCP_Parser(grammar, t)
            parser.set_input(t)
            parser.parse()
            self.assertTrue(parser.recognized())
            derivs = list(parser.all_derivation_trees())
            for der in derivs:
                print(der)
            self.assertEqual(1, len(derivs))
            parser.clear()


def constituent_tree_1():
    tree = ConstituentTree("s1")
    tree.add_leaf("f1", "VP", "hat")
    tree.add_leaf("f2", "ADV", "schnell")
    tree.add_leaf("f3", "VP", "gearbeitet")
    tree.add_punct("f4", "PUNC", ".")

    tree.set_label("V", "V")
    tree.add_child("V", "f1")
    tree.add_child("V", "f3")

    tree.set_label("ADV", "ADV")
    tree.add_child("ADV", "f2")

    tree.set_label("VP", "VP")
    tree.add_child("VP", "V")
    tree.add_child("VP", "ADV")

    tree.add_to_root("VP")

    return tree


def constituent_tree_2():
    tree = ConstituentTree("s2")
    tree.add_leaf("l1", "N", "John")
    tree.add_leaf("l2", "V", "hit")
    tree.add_leaf("l3", "D", "the")
    tree.add_leaf("l4", "N", "Ball")
    tree.add_punct("l5", "PUNC", ".")

    tree.set_label("NP", "NP")
    tree.add_child("NP", "l3")
    tree.add_child("NP", "l4")

    tree.set_label("VP", "VP")
    tree.add_child("VP", "l2")
    tree.add_child("VP", "NP")

    tree.set_label("S", "S")
    tree.add_child("S", "l1")
    tree.add_child("S", "VP")

    tree.add_to_root("S")

    return tree


def constituent_tree_1_pos_stripped():
    tree = ConstituentTree("s1")
    tree.add_leaf("f1", "--", "hat")
    tree.add_leaf("f2", "--", "schnell")
    tree.add_leaf("f3", "--", "gearbeitet")
    tree.add_punct("f4", "--", ".")

    tree.set_label("V", "V")
    tree.add_child("V", "f1")
    tree.add_child("V", "f3")

    tree.set_label("ADV", "ADV")
    tree.add_child("ADV", "f2")

    tree.set_label("VP", "VP")
    tree.add_child("VP", "V")
    tree.add_child("VP", "ADV")

    tree.add_to_root("VP")

    return tree


if __name__ == '__main__':
    unittest.main()
