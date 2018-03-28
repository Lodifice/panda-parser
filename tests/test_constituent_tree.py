from __future__ import print_function

__author__ = 'kilian'

import unittest
from hybridtree.constituent_tree import ConstituentTree
from constituent.induction import fringe_extract_lcfrs
from grammar.induction.recursive_partitioning import fanout_k_left_to_right, left_branching_partitioning
from collections import defaultdict
from constituent.construct_morph_annotation import build_nont_splits_dict, pos_cat_feats
from util.enumerator import Enumerator


class ConstituentTreeTest(unittest.TestCase):
    def test_something(self):
        tree = self.tree
        print("rooted", tree.root)
        tree.add_to_root("VP1")
        print("rooted", tree.root)

        print(tree)

        print("sent label", tree.sent_label())

        print("leaves", tree.leaves())

        print("is leaf (leaves)", [(x, tree.is_leaf(x)) for (x, _, _) in tree.leaves()])
        print("is leaf (internal)", [(x, tree.is_leaf(x)) for x in tree.ids()])
        print("leaf index", [(x, tree.leaf_index(x)) for x in ["f1", "f2", "f3"]])

        print("pos yield", tree.pos_yield())
        print("word yield", tree.word_yield())

        # reentrant
        # parent

        print("ids", tree.ids())

        # reorder
        print("n nodes", tree.n_nodes())
        print("n gaps", tree.n_gaps())

        print("fringe VP", tree.fringe("VP"))
        print("fringe V", tree.fringe("V"))

        print("empty fringe", tree.empty_fringe())

        print("complete?", tree.complete())

        print("max n spans", tree.max_n_spans())

        print("unlabelled structure", tree.unlabelled_structure())

        print("labelled spans", tree.labelled_spans())

    def test_induction(self):
        naming = 'child'

        def rec_part(tree):
            return left_branching_partitioning(len(tree.id_yield()))
            # return fanout_k_left_to_right(tree, 1)

        tree = self.tree
        tree.add_to_root("VP1")

        feature_log1 = defaultdict(lambda: 0)

        grammar = fringe_extract_lcfrs(tree, rec_part(tree), feature_logging=feature_log1, naming=naming)

        for key in feature_log1:
            print(key, feature_log1[key])

        print(grammar)

        feats = defaultdict(lambda: 0)
        grammar_ = fringe_extract_lcfrs(tree, rec_part(tree), isolate_pos=True, feature_logging=feats, naming=naming)

        print(grammar_)

        for key in feats:
            print(key, feats[key])

        print("Adding 2nd grammar to first")

        grammar.add_gram(grammar_, feature_logging=(feature_log1, feats))
        for idx in range(0, len(grammar.rules())):
            print(idx, grammar.rule_index(idx))

        print("Adding 3rd grammar to first")
        feats3 = defaultdict(lambda: 0)
        grammar3 = fringe_extract_lcfrs(self.tree2, rec_part(self.tree2), isolate_pos=True, feature_logging=feats3, naming=naming)
        grammar.add_gram(grammar3, feature_logging=(feature_log1, feats3))

        print()
        for idx in range(0, len(grammar.rules())):
            print(idx, grammar.rule_index(idx))
        print()
        print("New feature log")
        print()
        for key in feature_log1:
            print(key, feature_log1[key])
        grammar.make_proper()

        build_nont_splits_dict(grammar, feature_log1, nonterminals=Enumerator())

        print(grammar.rule_index(0))
        print(grammar.rule_index(2))

    def test_markovized_induction(self):
        naming = 'strict-markov-v-2-h-0'

        def rec_part(tree):
            return left_branching_partitioning(len(tree.id_yield()))
            # return fanout_k_left_to_right(tree, 1)

        tree = self.tree
        tree.add_to_root("VP1")

        print(tree)

        grammar = fringe_extract_lcfrs(tree, rec_part(tree), naming=naming, isolate_pos=True)
        print(grammar)

    def test_induction_2(self):
        def rec_part(tree):
            return left_branching_partitioning(len(tree.id_yield()))
        features = defaultdict(lambda: 0)
        grammar = fringe_extract_lcfrs(self.tree3, rec_part(self.tree3), naming="child", feature_logging=features, isolate_pos=True)
        grammar.make_proper()

        if False:
            for idx in range(0, len(grammar.rules())):
                print(grammar.rule_index(idx))
                for key in features:
                    if key[0] == idx:
                        print(key, features[key])
                print()
            for key in features:
                if type(key[0]) == int:
                    continue
                print(key, features[key])

        nont_splits, root_weights, rule_weights, _ = build_nont_splits_dict(grammar, features, nonterminals=Enumerator(), feat_function=pos_cat_feats, debug=True)
        print(nont_splits)
        print(root_weights)
        print(rule_weights)

    def setUp(self):
        tree = ConstituentTree("s1")
        tree.add_leaf("f1", "VAFIN", "hat", morph=[("number", "Sg"), ("person", "3"), ("tense", "Past")
            , ("mood", "Ind")])
        tree.add_leaf("f2", "ADV", "schnell", morph=[("degree", "Pos")])
        tree.add_leaf("f3", "VVPP", "gearbeitet")
        tree.add_punct("f4", "PUNC", ".")

        tree.add_child("VP2", "f1")
        tree.add_child("VP2", "f3")
        tree.add_child("ADVP", "f2")

        tree.add_child("VP1", "VP2")
        tree.add_child("VP1", "ADVP")

        tree.set_label("VP2", "VP")
        tree.set_label("VP1", "VP")
        tree.set_label("ADVP", "ADVP")

        self.tree = tree

        tree2 = ConstituentTree("s2")
        tree2.add_leaf("f1", "VAFIN", "haben", morph=[("number", "Pl"), ("person", "3"), ("tense", "Past"),
                                                      ("mood", "Ind")])
        tree2.add_leaf("f2", "ADV", "gut", morph=[("degree", "Pos")])
        tree2.add_leaf("f3", "VVPP", "gekocht")
        tree2.add_punct("f4", "PUNC", ".")

        tree2.add_child("VP2", "f1")
        tree2.add_child("VP2", "f3")
        tree2.add_child("ADVP", "f2")

        tree2.add_child("VP1", "VP2")
        tree2.add_child("VP1", "ADVP")

        tree2.set_label("VP2", "VP")
        tree2.set_label("VP1", "VP")
        tree2.set_label("ADVP", "ADVP")
        tree2.add_to_root("VP1")
        self.tree2 = tree2

        self.tree3 = ConstituentTree("s3")
        self.tree3.add_leaf("f1", "ADJA", "Allgemeiner", edge="NK", morph=[("number", "Sg")])
        self.tree3.add_leaf("f2", "ADJA", "Deutscher", edge="NK", morph=[("degree", "Pos"), ("number", "Sg")])
        self.tree3.add_leaf("f3", "NN", "Fahrrad", edge="NK", morph=[("number", "Sg"), ("gender", "Neut")])
        self.tree3.add_leaf("f4", "NN", "Club", edge="NK", morph=[("number", "Sg"), ("gender", "Neut")])
        for i in range(1,5):
            self.tree3.add_child("NP", "f" + str(i))
        self.tree3.set_label("NP", "NP")
        self.tree3.add_to_root("NP")

if __name__ == '__main__':
    unittest.main()
