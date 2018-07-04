#-*- coding: iso-8859-15 -*-
from __future__ import print_function
__author__ = 'kilian'

import copy
import sys
import unittest
from collections import defaultdict
from math import e
import corpora.negra_parse as np
import tempfile
import subprocess

from dependency.induction import induce_grammar
from grammar.induction.recursive_partitioning import left_branching, right_branching, cfg, \
    the_recursive_partitioning_factory, direct_extraction
from grammar.induction.terminal_labeling import the_terminal_labeling_factory
from dependency.labeling import the_labeling_factory
from grammar.linearization import linearize
from grammar.dcp import DCP_string
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.monadic_tokens import CoNLLToken, construct_conll_token, ConstituentCategory, ConstituentTerminal, construct_constituent_token
from parser.cpp_cfg_parser.parser_wrapper import CFGParser
from parser.discodop_parser.parser import DiscodopKbestParser
from grammar.lcfrs_derivation import derivation_to_hybrid_tree
from constituent.induction import fringe_extract_lcfrs
from itertools import product

try:
    from parser.fst.fst_export import compile_wfst_from_right_branching_grammar, fsa_from_list_of_symbols, compose, shortestpath, shortestdistance, retrieve_rules, PolishDerivation, ReversePolishDerivation, compile_wfst_from_left_branching_grammar, local_rule_stats, paths, LeftBranchingFSTParser
    test_pynini = True
except ModuleNotFoundError:
    test_pynini = False

from parser.sDCPevaluation.evaluator import DCP_evaluator, dcp_to_hybridtree
from parser.naive.parsing import LCFRS_parser
from tests.test_multiroot import multi_dep_tree
from grammar.induction.decomposition import fanout_limit_partitioning_with_guided_binarization, print_partitioning, n_spans
from grammar.lcfrs import LCFRS
from constituent.dummy_tree import flat_dummy_constituent_tree


class InductionTest(unittest.TestCase):
    def test_recursive_partitioning_transformation(self):
        tree = HybridTree("mytree")
        ids = ['a', 'b', 'c', 'd']
        for f in ids:
            tree.add_node(f, CoNLLToken(f, '_', '_', '_', '_', '_'), True, True)
            if f != 'a':
                tree.add_child('a', f)
        tree.add_to_root('a')

        print(tree)
        self.assertEqual([token.form() for token in tree.token_yield()], ids)
        self.assertEqual(tree.recursive_partitioning(), (set([0, 1, 2, 3]), [(set([0]), []), (set([1]), []), (set([2]), []), (set([3]), [])]))
        print(tree.recursive_partitioning())

        [fanout_1] = the_recursive_partitioning_factory().get_partitioning('fanout-1')

        print(fanout_1(tree))

    def test_single_root_induction(self):
        tree = hybrid_tree_1()
        # print tree.children("v")
        # print tree
        #
        # for id_set in ['v v1 v2 v21'.split(' '), 'v1 v2'.split(' '),
        # 'v v21'.split(' '), ['v'], ['v1'], ['v2'], ['v21']]:
        # print id_set, 'top:', top(tree, id_set), 'bottom:', bottom(tree, id_set)
        # print id_set, 'top_max:', max(tree, top(tree, id_set)), 'bottom_max:', max(tree, bottom(tree, id_set))
        #
        # print "some rule"
        # for mem, arg in [(-1, 0), (0,0), (1,0)]:
        # print create_DCP_rule(mem, arg, top_max(tree, ['v','v1','v2','v21']), bottom_max(tree, ['v','v1','v2','v21']),
        # [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1', 'v2'], ['v', 'v21']]])
        #
        #
        # print "some other rule"
        # for mem, arg in [(-1,1),(1,0)]:
        # print create_DCP_rule(mem, arg, top_max(tree, ['v1','v2']), bottom_max(tree, ['v1','v2']),
        # [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1'], ['v2']]])
        #
        # print 'strict:' , strict_labeling(tree, top_max(tree, ['v','v21']), bottom_max(tree, ['v','v21']))
        # print 'child:' , child_labeling(tree, top_max(tree, ['v','v21']), bottom_max(tree, ['v','v21']))
        # print '---'
        # print 'strict: ', strict_labeling(tree, top_max(tree, ['v1','v21']), bottom_max(tree, ['v1','v21']))
        # print 'child: ', child_labeling(tree, top_max(tree, ['v1','v21']), bottom_max(tree, ['v1','v21']))
        # print '---'
        # print 'strict:' , strict_labeling(tree, top_max(tree, ['v','v1', 'v21']), bottom_max(tree, ['v','v1', 'v21']))
        # print 'child:' , child_labeling(tree, top_max(tree, ['v','v1', 'v21']), bottom_max(tree, ['v','v1', 'v21']))

        tree2 = hybrid_tree_2()

        # print tree2.children("v")
        # print tree2
        #
        # print 'siblings v211', tree2.siblings('v211')
        # print top(tree2, ['v','v1', 'v211'])
        # print top_max(tree2, ['v','v1', 'v211'])
        #
        # print '---'
        # print 'strict:' , strict_labeling(tree2, top_max(tree2, ['v','v1', 'v211']), bottom_max(tree2, ['v','v11', 'v211']))
        # print 'child:' , child_labeling(tree2, top_max(tree2, ['v','v1', 'v211']), bottom_max(tree2, ['v','v11', 'v211']))

        # rec_par = ('v v1 v2 v21'.split(' '),
        # [('v1 v2'.split(' '), [(['v1'],[]), (['v2'],[])])
        #                ,('v v21'.split(' '), [(['v'],[]), (['v21'],[])])
        #            ])
        #
        # grammar = LCFRS(nonterminal_str(tree, top_max(tree, rec_par[0]), bottom_max(tree, rec_par[0]), 'strict'))
        #
        # add_rules_to_grammar_rec(tree, rec_par, grammar, 'child')
        #
        # grammar.make_proper()
        # print grammar

        print(tree.recursive_partitioning())

        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty','pos'),
                                      # the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel'),
                                      terminal_labeling.token_label, [direct_extraction], 'START')
        print(max([grammar.fanout(nont) for nont in grammar.nonts()]))
        print(grammar)

        parser = LCFRS_parser(grammar, 'NP N V V'.split(' '))
        print(parser.best_derivation_tree())

        tokens = [construct_conll_token(form, pos) for form, pos in
                  zip('Piet Marie helpen lezen'.split(' '), 'NP N V V'.split(' '))]
        hybrid_tree = HybridTree()
        hybrid_tree = parser.dcp_hybrid_tree_best_derivation(hybrid_tree, tokens, True,
                                                             construct_conll_token)
        print(list(map(str, hybrid_tree.full_token_yield())))
        print(hybrid_tree)

        string = "foo"
        dcp_string = DCP_string(string)
        dcp_string.set_edge_label("bar")
        print(dcp_string, dcp_string.edge_label())

        linearize(grammar, the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel'), the_terminal_labeling_factory().get_strategy('pos'), sys.stdout)

    def test_multiroot(self):
        tree = multi_dep_tree()
        term_pos = the_terminal_labeling_factory().get_strategy('pos').token_label
        fanout_1 = the_recursive_partitioning_factory().get_partitioning('fanout-1')
        for top_level_labeling_strategy in ['strict', 'child']:
            labeling_strategy = the_labeling_factory().create_simple_labeling_strategy(top_level_labeling_strategy,
                                                                                       'pos+deprel')
            for recursive_partitioning in [[direct_extraction], fanout_1, [left_branching]]:
                (_, grammar) = induce_grammar([tree], labeling_strategy, term_pos, recursive_partitioning, 'START')
                print(grammar)

                parser = LCFRS_parser(grammar, 'pA pB pC pD pE'.split(' '))
                print(parser.best_derivation_tree())

                cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                for token in cleaned_tokens:
                    token.set_edge_label('_')
                hybrid_tree = HybridTree()
                hybrid_tree = parser.dcp_hybrid_tree_best_derivation(hybrid_tree, cleaned_tokens, True,
                                                                     construct_conll_token)
                print(hybrid_tree)
                self.assertEqual(tree, hybrid_tree)

    def test_fst_compilation_right(self):
        if not test_pynini:
            return
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [right_branching], 'START')

        a, rules = compile_wfst_from_right_branching_grammar(grammar)

        print(repr(a))

        symboltable = a.input_symbols()

        string = 'NP N V V V'.split(' ')

        token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                          zip('Piet Marie helpen leren lezen'.split(' '), string)]


        fsa = fsa_from_list_of_symbols(string, symboltable)
        self.assertEqual('0\t1\tNP\tNP\n1\t2\tN\tN\n2\t3\tV\tV\n3\t4\tV\tV\n4\t5\tV\tV\n5\n', fsa.text().decode('utf-8'))

        b = compose(fsa, a)

        print(b.input_symbols())
        for i in b.input_symbols():
            print(i)


        print("Input Composition")
        print(b.text(symboltable, symboltable).decode('utf-8'))

        i = 0
        for path in paths(b):
            print(i, "th path:", path, end=' ')
            r = list(map(rules.index_object, path))
            d = PolishDerivation(r[1::])
            dcp = DCP_evaluator(d).getEvaluation()
            h = HybridTree()
            dcp_to_hybridtree(h, dcp, token_sequence, False, construct_conll_token)
            h.reorder()
            if h == tree2:
                print("correct")
            else:
                print("incorrect")
            i += 1

        stats = defaultdict(lambda: 0)
        local_rule_stats(b, stats, 15)

        print(stats)

        print("Shortest path probability")
        best = shortestpath(b)
        best.topsort()
        self.assertAlmostEqual(1.80844898756e-05, pow(e, -float(shortestdistance(best)[-1])))
        print(best.text())

        polish_rules = retrieve_rules(best)
        self.assertSequenceEqual(polish_rules, [8, 7, 1, 6, 2, 5, 3, 10, 3, 3])

        polish_rules = list(map(rules.index_object, polish_rules))

        print(polish_rules)

        der = PolishDerivation(polish_rules[1::])

        print(der)

        print(derivation_to_hybrid_tree(der, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

        dcp = DCP_evaluator(der).getEvaluation()

        h_tree_2 = HybridTree()
        dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                          construct_conll_token)

        print(h_tree_2)

    def test_fst_compilation_left(self):
        if not test_pynini:
            return
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [left_branching], 'START')

        fst, rules = compile_wfst_from_left_branching_grammar(grammar)

        print(repr(fst))

        symboltable = fst.input_symbols()

        string = ["NP", "N", "V", "V", "V"]

        fsa = fsa_from_list_of_symbols(string, symboltable)
        self.assertEqual(fsa.text().decode('utf-8'), '0\t1\tNP\tNP\n1\t2\tN\tN\n2\t3\tV\tV\n3\t4\tV\tV\n4\t5\tV\tV\n5\n')

        b = compose(fsa, fst)

        print(b.text(symboltable, symboltable))

        print("Shortest path probability", end=' ')
        best = shortestpath(b)
        best.topsort()
        # self.assertAlmostEquals(pow(e, -float(shortestdistance(best)[-1])), 1.80844898756e-05)
        print(best.text())

        polish_rules = retrieve_rules(best)
        self.assertSequenceEqual(polish_rules, [1, 2, 3, 4, 5, 4, 9, 4, 7, 8])

        polish_rules = list(map(rules.index_object, polish_rules))

        for rule in polish_rules:
            print(rule)
        print()

        der = ReversePolishDerivation(polish_rules[0:-1])
        self.assertTrue(der.check_integrity_recursive(der.root_id()))

        print(der)

        LeftBranchingFSTParser.preprocess_grammar(grammar)
        parser = LeftBranchingFSTParser(grammar, string)
        der_ = parser.best_derivation_tree()

        print(der_)
        self.assertTrue(der_.check_integrity_recursive(der_.root_id()))

        print(derivation_to_hybrid_tree(der, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

        print(derivation_to_hybrid_tree(der_, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

        dcp = DCP_evaluator(der).getEvaluation()

        h_tree_2 = HybridTree()
        token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                          zip('Piet Marie helpen lezen leren'.split(' '), 'NP N V V V'.split(' '))]
        dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                          construct_conll_token)

        print(h_tree_2)

    def test_cfg_parser(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for parser_class in [LCFRS_parser, CFGParser]:

            parser_class.preprocess_grammar(grammar)

            string = ["NP", "N", "V", "V", "V"]

            parser = parser_class(grammar, string)

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

    def test_generalization(self):
        tree = hybrid_tree_3()
        print(tree)
        print(list(map(lambda node: node.form(), tree.token_yield())))


        # rec_part = right_branching(tree)
        # rec_part_1 = cfg(tree)
        # print(rec_part_1)
        rec_part = the_recursive_partitioning_factory().get_partitioning('fanout-1-left-to-right')[0](tree)
        print(rec_part)
        # rec_part = left_branching(tree)
        # rec_part = cfg(tree)

        for naming in ['child', 'strict', 'strict-markov-v-1-h-0', 'strict-markov-v-1-h-1']:
            grammar = fringe_extract_lcfrs(tree, rec_part, naming=naming)

            if naming.startswith('strict-markov-v-1'):
                print(naming)
                print(grammar)

            parser = LCFRS_parser(grammar)

            for x in range(0, 10):
                sentence = ['ART'] + x * ['ADJA'] + ['NN']
                parser.set_input(sentence)
                parser.parse()

                # print(sentence, "Recognized:", parser.recognized())
                if naming == 'child':
                    rec = x > 0
                elif naming == 'strict':
                    rec = x == 3
                elif naming == 'strict-markov-v-1-h-1':
                    rec = x > 0
                else:
                    rec = True
                self.assertEqual(rec, parser.recognized())

                parser.clear()

            sentence = ['ART'] + ['NN'] + ['ADJA']
            parser.set_input(sentence)
            parser.parse()
            print(sentence, "Recognized:", parser.recognized())

    def test_generalization_2(self):
        tree = hybrid_tree_4()
        print(tree)
        print(list(map(lambda node: node.form(), tree.token_yield())))

        _, path = tempfile.mkstemp(suffix='.export')
        with open(path, 'w') as f:
            lines = np.serialize_hybridtrees_to_negra([tree], 1, 500)
            f.writelines(lines)

        _, path_bin = tempfile.mkstemp(suffix='.export')
        subprocess.call("discodop treetransforms --binarize --headrules".split() + ["util/negra.headrules", path, path_bin])

        corpus = np.sentence_names_to_hybridtrees(['1'], path_bin)

        tree_bin = corpus[0]
        print(tree_bin)

        rec_part = fanout_limit_partitioning_with_guided_binarization(direct_extraction(tree), 1, tree_bin)
        print(rec_part)

        subprocess.call(["rm", path])
        subprocess.call(["rm", path_bin])

        # rec_part = right_branching(tree)
        # rec_part_1 = cfg(tree)
        # print(rec_part_1)
        # rec_part = the_recursive_partitioning_factory().get_partitioning('fanout-1-left-to-right')[0](tree)
        # print(rec_part)
        # rec_part = left_branching(tree)
        # rec_part = cfg(tree)

        for naming in ['child', 'strict', 'strict-markov-v-1-h-0', 'strict-markov-v-1-h-1']:
            grammar = fringe_extract_lcfrs(tree, rec_part, naming=naming)

            if True or naming.startswith('strict-markov-v-1'):
                print(naming)
                print(grammar)

            parser = LCFRS_parser(grammar)

            for x, y in product(range(0, 5), range(0, 5)):
                sentence = ['ART'] + x * ['ADJA'] + ['NN'] + y * ['APPR', 'NN']
                parser.set_input(sentence)
                parser.parse()

                print(sentence, "Recognized:", parser.recognized())
                if naming == 'child':
                    rec = x >= 0 and y > 0
                elif naming == 'strict':
                    rec = x == 3 and y == 3
                elif naming == 'strict-markov-v-1-h-1':
                    rec = x > 0 and y > 0
                else:
                    rec = True
                self.assertEqual(rec, parser.recognized())

                parser.clear()

            sentence = ['ART'] + ['NN'] + ['ADJA']
            parser.set_input(sentence)
            parser.parse()
            print(sentence, "Recognized:", parser.recognized())

    def check_partitioning(self, partitioning, sent_length, fanout):
        r, _ = partitioning
        self.assertSetEqual(r, {i for i in range(sent_length)})

        def check_recursive(part, test):
            root, children = part
            if len(root) == 1:
                self.assertListEqual([], children)
                return
            test.assertLessEqual(n_spans(root), fanout)
            x = set()
            for i, c1 in enumerate(children):
                x = x.union(c1[0])
                for j, c2 in enumerate(children):
                    if i != j:
                        test.assertEqual(0, len(c1[0] & c2[0]))
            if not root == x:
                print(root)
                print(x)
                print(part)
            test.assertSetEqual(root, x)
            for c in children:
                check_recursive(c, test)

        check_recursive(partitioning, self)

    def test_induction_on_corpus(self):
        """
        Test the recursive partitioning transformation with guided binarization on a small corpus.
        """
        LIMIT = 500  # max is 5048
        CORPUS_PATH = 'res/TIGER/tiger21/tigerdev_root_attach.export'
        sent_ids = [str(i) for i in range(1, LIMIT * 10 + 2) if i % 10 == 1]

        PARSER = DiscodopKbestParser
        # PARSER = CFGParser  # todo CFGParser seems to be incomplete

        REC_PART = "guide-bin"
        # REC_PART = "fanout-1-left-to-right"
        cfg_left_to_right = the_recursive_partitioning_factory().get_partitioning('fanout-1-left-to-right')[0]

        # path_bin = '/tmp/tmpazyt5p3e.export'
        _, path_bin = tempfile.mkstemp(suffix='.export')
        subprocess.call(
            "discodop treetransforms --binarize --headrules".split() + ["util/negra.headrules", CORPUS_PATH, path_bin])

        corpus = np.sentence_names_to_hybridtrees(sent_ids, CORPUS_PATH, disconnect_punctuation=False, add_vroot=True)
        bin_corpus = np.sentence_names_to_hybridtrees(sent_ids, path_bin, disconnect_punctuation=False, add_vroot=True)

        grammar = LCFRS("START")
        FANOUT = 2
        NAMING = 'strict-markov-v-1-h-1'
        tree_counter = 0
        def f(token):
            return token.form()
        for tree, bin_tree in zip(corpus, bin_corpus):
            self.assertListEqual(list(map(f, tree.token_yield())), list(map(f, bin_tree.token_yield())))
            if REC_PART == "guide-bin":
                rec_part_direct = tree.recursive_partitioning()
                rec_part = fanout_limit_partitioning_with_guided_binarization(rec_part_direct, FANOUT, bin_tree)
            else:
                rec_part = cfg_left_to_right(tree)
            self.check_partitioning(rec_part, len(tree.id_yield()), FANOUT)

            try:
                tree_grammar = fringe_extract_lcfrs(tree, rec_part, NAMING, isolate_pos=False)

                tree_grammar_parser = PARSER(tree_grammar)
                tree_grammar_parser.set_input([token.pos() for token in tree.token_yield()])
                tree_grammar_parser.parse()
                if not tree_grammar_parser.recognized():
                    print(tree_grammar)
                    print(tree.sent_label())
                    print(tree)
                    print(tree_grammar_parser.input)
                    print(len(tree.id_yield()))
                    print_partitioning(rec_part)

                self.assertTrue(tree_grammar_parser.recognized())

                grammar.add_gram(tree_grammar)
            except IndexError:
                print(tree)
                print(bin_tree)
                print(rec_part)
            tree_counter += 1

        print("Rules", len(grammar.rules()), "Nonterminals", len(grammar.nonts()))
        print(tree_counter)
        grammar.make_proper()

        parser = PARSER(grammar)

        _, result_corpus = tempfile.mkstemp(suffix='.export')

        print("Parsing training sentences. Writing results to", result_corpus)

        with open(result_corpus, 'w') as rc:
            for tree in corpus:
                tokens = copy.deepcopy(tree.token_yield())
                parser.set_input([token.pos() for token in tokens])
                parser.parse()

                if not parser.recognized():
                    print(list(map(str, tokens)))
                    print(tree)

                self.assertTrue(parser.recognized())

                result_tree = ConstituentTree(tree.sent_label())

                result_tree = parser.dcp_hybrid_tree_best_derivation(result_tree, tokens, ignore_punctuation=False,
                                                                     construct_token=construct_constituent_token)
                self.assertTrue(result_tree is not None)

                result_tree.strip_vroot()
                rc.writelines(np.serialize_hybridtrees_to_negra([result_tree], 1, 500, use_sentence_names=True))
                parser.clear()

        _, result_corpus = tempfile.mkstemp(prefix='sys_', suffix='.export')
        _, gold_corpus = tempfile.mkstemp(prefix='gold_', suffix='.export')

        TEST_LIMIT = 100  # max is 5048
        sent_ids = [str(i) for i in range(LIMIT * 10 + 1, (LIMIT + TEST_LIMIT) * 10 + 2) if i % 10 == 1]
        corpus = np.sentence_names_to_hybridtrees(sent_ids, CORPUS_PATH, disconnect_punctuation=False, add_vroot=True)

        print("Parsing dev sentences. Writing results to", result_corpus)
        with open(result_corpus, 'w') as rc, open(gold_corpus, 'w') as gc:
            fails = 0
            for tree in corpus:
                tokens = copy.deepcopy(tree.token_yield())
                parser.set_input([token.pos() for token in tokens])
                parser.parse()

                if parser.recognized():
                    result_tree = ConstituentTree(tree.sent_label())

                    result_tree = parser.dcp_hybrid_tree_best_derivation(result_tree, tokens, ignore_punctuation=False,
                                                                         construct_token=construct_constituent_token)
                    self.assertTrue(result_tree is not None)

                else:
                    fails += 1
                    result_tree = flat_dummy_constituent_tree(tokens, tokens, None, 'VROOT', label=tree.sent_label())

                result_tree.strip_vroot()
                rc.writelines(np.serialize_hybridtrees_to_negra([result_tree], 1, 500, use_sentence_names=True))
                tree.strip_vroot()
                gc.writelines(np.serialize_hybridtrees_to_negra([tree], 1, 500, use_sentence_names=True))
                parser.clear()
            print("Parse failures:", fails)

        subprocess.call(["discodop", "eval", gold_corpus, result_corpus, "util/proper.prm"])


def hybrid_tree_1():
    tree = HybridTree()
    tree.add_node('v1', CoNLLToken('Piet', '_', 'NP', 'NP', '_', 'SBJ'), True)
    tree.add_node('v21', CoNLLToken('Marie', '_', 'N', 'N', '_', 'OBJ'), True)
    tree.add_node('v', CoNLLToken('helpen', '_', 'V', 'V', '_', 'ROOT'), True)
    tree.add_node('v2', CoNLLToken('lezen', '_', 'V', 'V', '_', 'VBI'), True)
    tree.add_child('v', 'v2')
    tree.add_child('v', 'v1')
    tree.add_child('v2', 'v21')
    tree.add_to_root('v')
    tree.reorder()
    return tree


def hybrid_tree_2():
    tree2 = HybridTree()
    tree2.add_node('v1', CoNLLToken('Piet', '_', 'NP', 'NP', '_', 'SBJ'), True)
    tree2.add_node('v211', CoNLLToken('Marie', '_', 'N', 'N', '_', 'OBJ'), True)
    tree2.add_node('v', CoNLLToken('helpen', '_', 'V', 'V', '_', 'ROOT'), True)
    tree2.add_node('v2', CoNLLToken('leren', '_', 'V', 'V', '_', 'VBI'), True)
    tree2.add_node('v21', CoNLLToken('lezen', '_', 'V', 'V', '_', 'VFIN'), True)
    tree2.add_child('v', 'v2')
    tree2.add_child('v', 'v1')
    tree2.add_child('v2', 'v21')
    tree2.add_child('v21', 'v211')
    tree2.add_to_root('v')
    tree2.reorder()
    return tree2


def hybrid_tree_3():
    tree = ConstituentTree()
    tree.add_node('v', ConstituentCategory('NP'))
    tree.add_node('v1', ConstituentTerminal('eine', 'ART'), order=True)
    tree.add_node('v3', ConstituentTerminal('kluge', 'ADJA'), order=True)
    tree.add_node('v4', ConstituentTerminal('glückliche', 'ADJA'), order=True)
    tree.add_node('v5', ConstituentTerminal('schöne', 'ADJA'), order=True)
    tree.add_node('v2', ConstituentTerminal('Frau', 'NN'), order=True)
    for i in range(1, 6):
        tree.add_child('v', 'v' + str(i))
    tree.add_to_root('v')
    tree.reorder()
    return tree


def hybrid_tree_4():
    tree = hybrid_tree_3()
    tree.add_node('v6', ConstituentCategory('PP'))
    tree.add_node('v61', ConstituentTerminal('mit', 'APPR'), order=True)
    tree.add_node('v62', ConstituentTerminal('Verantwortung', 'NN'), order=True)
    tree.add_child('v', 'v6')
    tree.add_child('v6', 'v61')
    tree.add_child('v6', 'v62')
    tree.add_node('v7', ConstituentCategory('PP'))
    tree.add_node('v71', ConstituentTerminal('aus', 'APPR'), order=True)
    tree.add_node('v72', ConstituentTerminal('Deutschland', 'NN'), order=True)
    tree.add_child('v', 'v7')
    tree.add_child('v7', 'v71')
    tree.add_child('v7', 'v72')
    tree.add_node('v8', ConstituentCategory('PP'))
    tree.add_node('v81', ConstituentTerminal('ohne', 'APPR'), order=True)
    tree.add_node('v82', ConstituentTerminal('Angst', 'NN'), order=True)
    tree.add_child('v', 'v8')
    tree.add_child('v8', 'v81')
    tree.add_child('v8', 'v82')
    return tree


if __name__ == '__main__':
    unittest.main()