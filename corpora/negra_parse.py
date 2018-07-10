"""Parsing/Serialization from and to the Negra export format into HybridTrees, HybridDags, and (only to)
DeepSyntaxGraphs."""
from __future__ import print_function, unicode_literals

import codecs
import os
import re
from collections import defaultdict
from os.path import expanduser
from typing import Iterable

from graphs.dog import DeepSyntaxGraph
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.general_hybrid_tree import HybridDag
from hybridtree.monadic_tokens import ConstituentTerminal, ConstituentCategory
from util.enumerator import Enumerator

# Used only by CL experiments
# Location of Negra corpus.
NEGRA_DIRECTORY = 'res/negra-corpus/downloadv2'
# The non-projective and projective versions of the negra corpus.
NEGRA_NONPROJECTIVE = os.path.join(NEGRA_DIRECTORY, '/negra-corpus.export')
NEGRA_PROJECTIVE = os.path.join(NEGRA_DIRECTORY, '/negra-corpus.cfg')


DISCODOP_HEADER = re.compile(r'^%%\s+word\s+lemma\s+tag\s+morph\s+edge\s+parent\s+secedge$')
BOS = re.compile(r'^#BOS\s+([0-9]+)')
EOS = re.compile(r'^#EOS\s+([0-9]+)$')

STANDARD_NONTERMINAL = re.compile(r'^#([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')
STANDARD_TERMINAL = re.compile(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')

DISCODOP_NONTERMINAL = re.compile(r'^#([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+'
                                  r'([^\s]+)\s+([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')
DISCODOP_TERMINAL = re.compile(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+'
                               r'([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')


def num_to_name(num):
    """
    :type num: int
    :rtype: str
    convert sentence number to name
    """
    return str(num)


def sentence_names_to_hybridtrees(names,
                                  path,
                                  enc="utf-8",
                                  disconnect_punctuation=True,
                                  add_vroot=False,
                                  mode="STANDARD",
                                  secedge=False):
    """
    :param names:  list of sentence identifiers
    :type names: Iterable[str]
    :param path: path to corpus
    :type path: str
    :param enc: file encoding
    :type enc: str
    :param disconnect_punctuation: disconnect
    :type disconnect_punctuation: bool
    :param add_vroot: adds a virtual root node labelled 'VROOT'
    :type add_vroot: bool
    :param mode: either 'STANDARD' (no lemma field) or 'DISCODOP' (lemma field)
    :type mode: str
    :param secedge: add secondary edges
    :type secedge: bool
    :return: list of constituent structures (HybridTrees or HybridDags) from file_name whose names are in names
    """
    negra = codecs.open(expanduser(path), encoding=enc)
    trees = []
    tree = None
    name = ''
    n_leaves = 0
    for line in negra:
        match_mode = DISCODOP_HEADER.match(line)
        if match_mode:
            mode = "DISCO-DOP"
            continue
        match_sent_start = BOS.search(line)
        match_sent_end = EOS.match(line)
        if mode == "STANDARD":
            match_nont = \
                STANDARD_NONTERMINAL.match(line)
            match_term = \
                STANDARD_TERMINAL.match(line)
        elif mode == "DISCO-DOP":
            match_nont = DISCODOP_NONTERMINAL.match(line)
            match_term = DISCODOP_TERMINAL.match(line)
        if match_sent_start:
            this_name = match_sent_start.group(1)
            if this_name in names:
                name = this_name
                if secedge:
                    tree = HybridDag(name)
                else:
                    tree = ConstituentTree(name)
                n_leaves = 0
                if add_vroot:
                    tree.set_label('0', 'VROOT')
                    tree.add_to_root('0')
        elif match_sent_end:
            this_name = match_sent_end.group(1)
            if name == this_name:
                tree.reorder()
                trees += [tree]
                tree = None
        elif tree:
            if match_nont:
                id = match_nont.group(1)
                if mode == "STANDARD":
                    OFFSET = 0
                else:
                    OFFSET = 1
                nont = match_nont.group(2 + OFFSET)
                edge = match_nont.group(4 + OFFSET)
                parent = match_nont.group(5 + OFFSET)
                # print(match_nont.groups(), len(match_nont.groups()))
                secedges = [] if not secedge or match_nont.group(6 + OFFSET) is None else \
                    match_nont.group(6 + OFFSET).split()

                tree.add_node(id, ConstituentCategory(nont), False, True)

                tree.node_token(id).set_edge_label(edge)
                if parent == '0' and not add_vroot:
                    tree.add_to_root(id)
                else:
                    tree.add_child(parent, id)
                if secedge and secedges:
                    # print(secedges)
                    for sei in range(0, len(secedges) // 2, 2):
                        sec_label = secedges[sei]
                        sec_parent = secedges[sei + 1]
                        tree.add_sec_child(sec_parent, id, sec_label)
            elif match_term:
                if mode == "STANDARD":
                    OFFSET = 0
                else:
                    OFFSET = 1

                word = match_term.group(1)
                pos = match_term.group(2 + OFFSET)
                edge = match_term.group(4 + OFFSET)
                parent = match_term.group(5 + OFFSET)
                # print(match_term.groups(), len(match_term.groups()))
                secedges = [] if not secedge or match_term.group(6 + OFFSET) is None else \
                    match_term.group(6 + OFFSET).split()

                n_leaves += 1
                leaf_id = str(100 + n_leaves)
                if parent == '0' and disconnect_punctuation:
                    tree.add_punct(leaf_id, pos, word)
                else:
                    if parent == '0' and not add_vroot:
                        tree.add_to_root(leaf_id)
                    else:
                        tree.add_child(parent, leaf_id)

                    token = ConstituentTerminal(word, pos, edge, None, '--')
                    tree.add_node(leaf_id, token, True, True)

                    tree.node_token(leaf_id).set_edge_label(edge)
                    if secedge and secedges:
                        # print(secedges)
                        for sei in range(0, len(secedges) // 2, 2):
                            sec_label = secedges[sei]
                            # assert secedges[sei] == edge
                            sec_parent = secedges[sei + 1]
                            tree.add_sec_child(sec_parent, leaf_id, sec_label)
    negra.close()
    return trees


def topological_order(dag):
    """
    :type dag: HybridDag
    :return: list of nodes of dag in topological order
    :rtype: list
    """
    order = []
    added = set()
    changed = True
    while changed:
        changed = False
        for node in dag.nodes():
            if node in added:
                continue
            if all([c in added for c in dag.children(node) + dag.sec_children(node)]):
                added.add(node)
                order.append(node)
                changed = True
    assert len(added) == len(dag.nodes())
    # print("Order", order)
    return order


def generate_ids_for_inner_nodes_dag(dag, order, idNum):
    counter = 500
    for node in order:
        if node not in dag.full_yield():
            idNum[node] = counter
            counter += 1


def generate_ids_for_inner_nodes(tree, node_id, idNum):
    """
    generates a dictionary which assigns each tree id an numeric id as required by export format
    :param tree: parse tree
    :type: ConstituentTree
    :param node_id: id of current node
    :type: str
    :param idNum: current dictionary 
    :type: dict
    :return: nothing
    """
    count = 500+len([n for n in tree.nodes() if n not in tree.full_yield()])

    if len(idNum) is not 0:
        count = min(idNum.values())

    if node_id not in tree.id_yield():
        idNum[node_id] = count-1

    for child in tree.children(node_id):
        generate_ids_for_inner_nodes(tree, child, idNum)

    return


def hybridtree_to_sentence_name(tree, idNum):
    """
    generates lines for given tree in export format
    :param tree: parse tree
    :type: ConstituentTree
    :param idNum: dictionary mapping node id to a numeric id
    :type: dict
    :return: list of lines
    :rtype: list of str
    """
    lines = []

    for leaf in tree.full_yield():
        token = tree.node_token(leaf)
        morph = u'--'
        # if not isinstance(token.form(), str):
        #     print(token.form(), type(token.form()))
        #     assert isinstance(token.form(), str)
        line = [token.form(), token.pos(), morph, token.edge()]

        # special handling of disconnected punctuation
        if leaf in tree.id_yield() and leaf not in tree.root:
            if tree.parent(leaf) is None or tree.parent(leaf) not in idNum:
                print(tree, leaf, tree.full_yield(), list(map(str, tree.full_token_yield())), tree.parent(leaf),
                      tree.parent(leaf) in idNum)
                assert False and "Words (i.e. leaves) should not have secondary children!"
            line.append(str(idNum[tree.parent(leaf)]))
        else:
            line.append(u'0')

        if isinstance(tree, HybridDag):
            for p in tree.sec_parents(leaf):
                line.append(token.edge())
                line.append(str(idNum[p]))

        lines.append(u'\t'.join(line) + u'\n')

    category_lines = []

    for node in [n for n in tree.nodes() if n not in tree.full_yield()]:
        token = tree.node_token(node)
        morph = u'--'

        line = [u'#' + str(idNum[node]), str(token.category()), morph, token.edge()]

        if node in tree.root:
            line.append(u'0')
        elif node not in tree.root:
            line.append(str(idNum[tree.parent(node)]))

        if isinstance(tree, HybridDag):
            for p in tree.sec_parents(node):
                line.append(token.edge())
                line.append(str(idNum[p]))

        category_lines.append(line)

    for line in sorted(category_lines, key=lambda l: l[0]):
        lines.append(u'\t'.join(line) + u'\n')

    return lines


def serialize_hybridtrees_to_negra(trees, counter, length, use_sentence_names=False):
    """
    converts a sequence of parse tree to the negra export format
    :param trees: list of parse trees
    :type: list of ConstituentTrees
    :return: list of export format lines
    :rtype: list of str
    """
    sentence_names = []

    for tree in trees:
        if len(tree.full_yield()) <= length:
            idNum = {}
            if isinstance(tree, HybridDag):
                generate_ids_for_inner_nodes_dag(tree, topological_order(tree), idNum)
                print(idNum)
            else:
                for root in tree.root:
                    generate_ids_for_inner_nodes(tree, root, idNum)
            if use_sentence_names:
                s_name = str(tree.sent_label())
            else:
                s_name = str(counter)
            sentence_names.append(u'#BOS ' + s_name + u'\n')
            sentence_names.extend(hybridtree_to_sentence_name(tree, idNum))
            sentence_names.append(u'#EOS ' + s_name + u'\n')
            counter += 1

    return sentence_names


def serialize_acyclic_dogs_to_negra(dsg, sec_edge_to_terminal=False):
    """
    converts a sequence of acyclic syntax graphs to the negra export format
    :type dsg: DeepSyntaxGraph
    :type sec_edge_to_terminal: bool
    :param sec_edge_to_terminal: if true, exports secondary edges with terminals as target
    """
    assert not dsg.dog.cyclic()
    assert len(dsg.sentence) < 500

    enum = Enumerator(first_index=500)
    # NB: contrary to the export standard, we index words starting from 1 (and not starting from 0)
    # NB: because 0 also refers to the virtual root (important for sec_edge_to_terminal == True)
    # NB: see http://www.coli.uni-saarland.de/~thorsten/publications/Brants-CLAUS98.pdf
    # NB: only relevant for TiGer s22084, probably annotation error
    synced_idxs = {idx: i + 1 for i, l in enumerate(dsg.synchronization) for idx in l}

    def idNum(tree_idx):
        if tree_idx in synced_idxs:
            return str(synced_idxs[tree_idx])
        else:
            return str(enum.object_index(tree_idx))

    # NB: here we enforce the indices to be topologically ordered as required by the export standard
    for idx in dsg.dog.topological_order():
        if idx not in synced_idxs:
            idNum(idx)

    lines = []

    for idx, token in enumerate(dsg.sentence):
        assert isinstance(token, ConstituentTerminal)
        # if not isinstance(token.form(), str):
        #     print(token.form(), type(token.form()))
        #     assert isinstance(token.form(), str)
        morph_order = ['person', 'case', 'number', 'tense', 'mood', 'gender', 'degree']
        morph = sorted(token.morph_feats(), key=lambda x: morph_order.index(x[0]))
        morph = '.'.join([str(x[1]) for x in morph if str(x[1]) != '--'])
        if morph == '':
            morph = u'--'
        line = [token.form(), token.pos(), morph]
        tree_idx = dsg.get_graph_position(idx)
        assert len(tree_idx) == 1
        tree_idx = tree_idx[0]

        parents = []
        if tree_idx in dsg.dog.outputs:
            parents.append(u'--')
            parents.append(u'0')

        for parent_idx in dsg.dog.parents:
            if not sec_edge_to_terminal and parent_idx in synced_idxs:
                continue
            edge = dsg.dog.incoming_edge(parent_idx)
            for j, child_idx in enumerate(edge.inputs):
                if child_idx == tree_idx:
                    if j in edge.primary_inputs:
                        parents = [edge.get_function(j), idNum(parent_idx)] + parents
                    else:
                        parents.append(edge.get_function(j))
                        parents.append(idNum(parent_idx))
        line += parents
        lines.append(u'\t'.join(line) + u'\n')

    category_lines = []
    for tree_idx in dsg.dog.nodes:
        token = dsg.dog.incoming_edge(tree_idx).label
        if isinstance(token, ConstituentTerminal):
            continue
        morph = u'--'

        line = ['#' + str(idNum(tree_idx)), token, morph]

        parents = []
        if tree_idx in dsg.dog.outputs:
            parents.append(u'--')
            parents.append(u'0')

        for parent_idx in dsg.dog.parents:
            if not sec_edge_to_terminal and parent_idx in synced_idxs:
                continue
            edge = dsg.dog.incoming_edge(parent_idx)
            for j, child_idx in enumerate(edge.inputs):
                if child_idx == tree_idx:
                    if j in edge.primary_inputs:
                        parents = [edge.get_function(j), idNum(parent_idx)] + parents
                    else:
                        parents.append(edge.get_function(j))
                        parents.append(idNum(parent_idx))
        line += parents

        category_lines.append(line)

    category_lines = sorted(category_lines, key=lambda x: x[0])

    for line in category_lines:
        lines.append(u'\t'.join(line) + u'\n')

    return lines


def serialize_hybrid_dag_to_negra(dsgs, counter, length, use_sentence_names=False):
    """
    converts a sequence of parse tree to the export format
    :param dsgs: list of parse trees
    :type dsgs: list[DeepSyntaxGraph]
    :return: list of export format lines
    :rtype: list of str
    """
    sentence_names = []

    for dsg in dsgs:
        if len(dsg.sentence) <= length:
            if use_sentence_names:
                name = str(dsg.label)
            else:
                name = str(counter)
            sentence_names.append(u'#BOS ' + name + u'\n')
            sentence_names.extend(serialize_acyclic_dogs_to_negra(dsg))
            sentence_names.append(u'#EOS ' + name + u'\n')
            counter += 1

    return sentence_names


def negra_to_json(dsg, terminal_encoding, terminal_labeling, delimiter=' : '):
    """
    :param dsg:
    :type dsg: HybridDag
    :return:
    :rtype:
    """

    node_io = {}

    def dag_to_json():
        data = {"type": "hypergraph"}
        data['nodes'] = []
        data['edges'] = []

        edge_idx = 0
        next_node_idx = 0
        sec_children = defaultdict(lambda: [])

        def dag_to_json_rec(node, node_idx):
            nonlocal next_node_idx
            nonlocal edge_idx
            nonlocal node_io
            output = next_node_idx
            next_node_idx += 1
            data['nodes'].append(output)
            first_child = next_node_idx
            data['nodes'].append(first_child)
            next_child = first_child
            next_node_idx += 1
            for child in dsg.children(node):
                next_child = dag_to_json_rec(child, next_child)
            for _ in dsg.sec_children(node):
                c_output = next_node_idx
                next_node_idx += 1
                data['nodes'].append(c_output)
                sec_children[node].append((next_child, c_output))
                next_child = c_output

            token = dsg.node_token(node)
            label = terminal_labeling.token_tree_label(token)
            label = terminal_encoding.object_index(label)
            node_io[node] = node_idx, output
            data['edges'].append({'id': edge_idx,
                                  'label': label,
                                  # only first and last child
                                  'attachment': [first_child, next_child, node_idx, output],
                                  'terminal': True})

            edge_idx += 1
            return output

        def add_sec_children(node):
            assert len(dsg.sec_children(node)) == len(sec_children[node])
            nonlocal edge_idx
            for child, (inp, outp), edge_label in zip(dsg.sec_children(node), sec_children[node], dsg.sec_child_edge_labels(node)):
                ci, co = node_io[child]
                label = 'SECEDGE' + delimiter + edge_label
                label = terminal_encoding.object_index(label)
                data['edges'].append({'id': edge_idx,
                                      'label': label,
                                      'attachment': [ci, co, inp, outp],
                                      'terminal': True
                                      })
                edge_idx += 1

        first_root = next_node_idx
        data['nodes'].append(first_root)
        next_node_idx += 1
        next_root = first_root
        for root in dsg.root:
            next_root = dag_to_json_rec(root, next_root)
        for node in dsg.nodes():
            add_sec_children(node)

        data['ports'] = [first_root, next_root]
        return data

    def string_to_graph_json(token_sequence, start_node, start_edge):
        return {'type': 'hypergraph',
                'nodes': [i for i in range(start_node, start_node + len(token_sequence) + 1)],
                'edges': [{'id': idx + start_edge,
                           'label': terminal_encoding.object_index(terminal_labeling.token_label(token)),
                           'attachment': [start_node + idx, start_node + idx + 1],
                           'terminal': True}
                          for idx, token in enumerate(token_sequence)],
                'ports': [start_node, start_node + len(token_sequence)]
                }

    data = {"type": "bihypergraph"}
    data["G1"] = dag_to_json()
    # print(data["G2"])
    max_node = max(data["G1"]['nodes'])
    max_edge = max(map(lambda x: x['id'], data["G1"]['edges']))
    data["G2"] = string_to_graph_json(dsg.token_yield(), start_node=max_node + 1, start_edge=max_edge + 1)
    max_edge = max(map(lambda x: x['id'], data["G2"]['edges']))
    data["alignment"] = [{'id': j + max_edge + 1,
                          'label': -1,
                           'attachment': [node_io[idx][0], max_node + 1 + j]}
                           for j, idx in enumerate(dsg.id_yield())]
    return data


def export_corpus_to_json(corpus, terminals, terminal_labeling=str, delimiter=' : '):
    data = {"corpus": [],
            "alignmentLabel": terminals.object_index(None),
            "nonterminalEdgeLabel": terminals.object_index(None)
            }
    for dsg in corpus:
        data["corpus"].append(negra_to_json(dsg, terminals, terminal_labeling=terminal_labeling, delimiter=delimiter))
    return data


__all__ = ["sentence_names_to_hybridtrees", "serialize_hybridtrees_to_negra", "hybridtree_to_sentence_name",
           "serialize_acyclic_dogs_to_negra", "serialize_hybrid_dag_to_negra", "negra_to_json", "export_corpus_to_json"]
