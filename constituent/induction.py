# coding: utf-8
from __future__ import print_function
# Extracting grammars out of hybrid trees.

# from hybridtree import *
from grammar.induction.decomposition import *
from grammar.lcfrs import *
from grammar.dcp import *
from grammar.induction.terminal_labeling import PosTerminals
from hybridtree.constituent_tree import ConstituentTree
import copy
import re
# The root symbol.
START = 'START'


class BasicNonterminalLabeling:
    def label_nont(self, tree, id):
        token = tree.node_token(id)
        if token.type() == "CONSTITUENT-CATEGORY":
            return token.category()
        elif token.type() == "CONSTITUENT-TERMINAL":
            return token.pos()


class NonterminalsWithFunctions:
    def label_nont(self, tree, id):
        token = tree.node_token(id)
        if token.type() == "CONSTITUENT-CATEGORY":
            return token.category() + "/" + token.edge()
        elif token.type() == "CONSTITUENT-TERMINAL":
            return token.pos() + "/" + token.edge()


def direct_extract_lcfrs(tree,
                         term_labeling=PosTerminals(),
                         nont_labeling=BasicNonterminalLabeling(),
                         binarize=False,
                         isolate_pos=False,
                         hmarkov=0):
    """
    :type tree: ConstituentTree
    :type term_labeling: ConstituentTerminalLabeling
    :type binarize: bool
    :type isolate_pos: bool
    :type hmarkov: int
    :rtype: LCFRS
    Extract LCFRS directly from hybrid tree.
    """
    assert not binarize or isolate_pos

    # Binarization without POS isolation requires a more sophisticated sDCP handling
    # with more than one variable per nonterminal. This is not implemented.
    # see, e.g., TIGER sentence 328:
    #
    #                                             ROOT
    # ┌──────────────────────┬─────────────────────┼────────────────────────────────────────────────────────────────┐
    # │                      │                     S                                                                │
    # │                      │    ┌────┬───┬───────┴───────────┬──────┬──────────────────────────────────┐          │
    # │                      │    │    NP  │                   │      │                                  │          │
    # │     ┌──────┬────┬─── │ ── │ ───┴── │ ───────────────── │ ──── │ ────────────┐                    │          │
    # │     │      │    │    │    │        NP                  │      │             PP                   PP         │
    # │     │      │    │    │    │    ┌───┴───────┐           │      │      ┌──────┼──────┐       ┌─────┴────┐     │
    # $[   ADV    ADV  CARD  $[ VAFIN ART          NN         ADV    ADV    APPR   CARD    NN   APPRART       NN    $.
    # │     │      │    │    │    │    │           │           │      │      │      │      │       │          │     │
    # `` Deutlich über 5000  ''  hat  die   SPD-Stadtregieru jetzt jeweils binnen zwölf Monaten    im       Visier  .
    #                                              ng
    #
    # [a] S/1(LCFRS_var(mem=0, arg=0) LCFRS_var(mem=1, arg=0) ADV ADV LCFRS_var(mem=0, arg=1) LCFRS_var(mem=1, arg=1))
    #     -> NP/2 BAR/S/2		<0>=S:{--}(<0,0> <1,0> [0:{MO}]() [1:{MO}]())
    # [b] BAR/S/2(VAFIN LCFRS_var(mem=0, arg=0); LCFRS_var(mem=1, arg=0))
    #     -> NP/1 PP/1		<0>=[0:{HD}]() <0,0> <1,0>
    # here the <1,0> variable in rule [a] needs to be split into <1,0> and <1,1>, because the NP occurs before "ADV ADV"
    # in the canonically ordered tree, but PP occurs afterwards
    #
    # TODO: Claim: This only affects BAR/.. nonterminals. Not more than fanout(BAR/...) nonterminals are needed.
    # TODO: Thus, each BAR/.. nonterminal gets a uniform number of sDCP arguments, some may be empty.
    # TODO: The term.args() string can be analyzed to construct appropriate sDCP rules for the binarization artifacts.

    gram = LCFRS(start=START)
    root = tree.root[0]
    if tree.is_leaf(root):
        lhs = LCFRS_lhs(START)
        label = term_labeling.token_label(tree.node_token(root))
        lhs.add_arg([label])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_term(DCP_index(0, edge_label=tree.node_token(root).edge()), [])])
        gram.add_rule(lhs, [], dcp=[dcp_rule])
    else:
        first = direct_extract_lcfrs_from(tree, root, gram, term_labeling, nont_labeling, binarize, isolate_pos,
                                          hmarkov=hmarkov)
        lhs = LCFRS_lhs(START)
        lhs.add_arg([LCFRS_var(0, 0)])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_var(0, 0)])
        gram.add_rule(lhs, [first], dcp=[dcp_rule])
    return gram


def direct_extract_lcfrs_from(tree,
                              id,
                              gram,
                              term_labeling,
                              nont_labeling,
                              binarization,
                              isolate_pos=False,
                              hmarkov=0):
    """
    :type tree: ConstituentTree
    :type id: str
    :type gram: LCFRS
    :type term_labeling: ConstituentTerminalLabeling
    :type isolate_pos: bool
    :type binarization: bool
    :type hmarkov: int
    :rtype: str

    Traverse subtree at id and put extracted rules in grammar.
    """
    fringe = tree.fringe(id)
    spans = join_spans(fringe)
    nont_fanout = len(spans)
    nont = nont_labeling.label_nont(tree, id) + '/' + str(nont_fanout)
    lhs = LCFRS_lhs(nont)

    if tree.is_leaf(id):
        label = term_labeling.token_label(tree.node_token(id))
        lhs.add_arg([label])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_term(DCP_index(0, edge_label=tree.node_token(id).edge()), [])])
        gram.add_rule(lhs, [], dcp=[dcp_rule])
        return lhs.nont()

    children = [(child, join_spans(tree.fringe(child))) \
                for child in tree.children(id)]
    edge_labels = []
    for (low, high) in spans:
        arg = []
        pos = low
        while pos <= high:
            child_num = 0
            for i, (child, child_spans) in enumerate(children):
                for j, (child_low, child_high) in enumerate(child_spans):
                    if pos == child_low:
                        if tree.is_leaf(child) and not isolate_pos:
                            arg += [term_labeling.token_label(tree.node_token(child))]
                            edge_labels += [tree.node_token(child).edge()]
                        else:
                            arg += [LCFRS_var(child_num, j)]
                        pos = child_high + 1
                if not tree.is_leaf(child) or isolate_pos:
                    child_num += 1
        lhs.add_arg(arg)

    dcp_term_args = []
    rhs = []
    nont_counter = 0
    term_counter = 0
    for (child, child_spans) in children:
        if not tree.is_leaf(child) or isolate_pos:
            rhs_nont_fanout = len(child_spans)
            rhs += [nont_labeling.label_nont(tree, child) + '/' + str(rhs_nont_fanout)]
            dcp_term_args.append(DCP_var(nont_counter, 0))
            nont_counter += 1
        else:
            dcp_term_args.append(DCP_term(DCP_index(term_counter, edge_label=edge_labels[term_counter]), []))
            term_counter += 1

    dcp_lhs = DCP_var(-1, 0)
    # dcp_indices = [DCP_term(DCP_index(i, edge_label=edge), []) for i, edge in enumerate(edge_labels)]
    # dcp_vars = [DCP_var(i, 0) for i in range(len(rhs))]
    # dcp_term_args = dcp_indices + dcp_vars
    label = tree.node_token(id).category()
    dcp_term = DCP_term(DCP_string(label, edge_label=tree.node_token(id).edge()), dcp_term_args)
    dcp_rule = [DCP_rule(dcp_lhs, [dcp_term])]
    if binarization:
        for lhs_, rhs_, dcp_ in binarize(lhs, rhs, dcp_rule, hmarkov=hmarkov):
            gram.add_rule(lhs_, rhs_, dcp=dcp_)
    else:
        gram.add_rule(lhs, rhs, dcp=dcp_rule)
    for (child, _) in children:
        if not tree.is_leaf(child) or isolate_pos:
            direct_extract_lcfrs_from(tree, child, gram, term_labeling, nont_labeling, binarization, isolate_pos,
                                      hmarkov=hmarkov)
    return nont


def direct_extract_lcfrs_from_prebinarized_corpus(tree,
                                                  term_labeling=PosTerminals(),
                                                  nont_labeling=BasicNonterminalLabeling(),
                                                  isolate_pos=True):
    gram = LCFRS(start=START)
    root = tree.root[0]
    if tree.is_leaf(root):
        lhs = LCFRS_lhs(START)
        label = term_labeling.token_label(tree.node_token(root))
        lhs.add_arg([label])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_term(DCP_index(0, edge_label=tree.node_token(root).edge()), [])])
        gram.add_rule(lhs, [], dcp=[dcp_rule])
    else:
        first = direct_extract_lcfrs_prebinarized_recur(tree, root, gram, term_labeling, nont_labeling, isolate_pos)
        lhs = LCFRS_lhs(START)
        lhs.add_arg([LCFRS_var(0, 0)])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_var(0, 0)])
        gram.add_rule(lhs, [first], dcp=[dcp_rule])
    return gram


def direct_extract_lcfrs_prebinarized_recur(tree,
                                            idx,
                                            gram,
                                            term_labeling,
                                            nont_labeling,
                                            isolate_pos):
    fringe = tree.fringe(idx)
    spans = join_spans(fringe)
    nont_fanout = len(spans)
    nont = nont_labeling.label_nont(tree, idx) + '/' + str(nont_fanout)

    lhs = LCFRS_lhs(nont)

    if tree.is_leaf(idx):
        label = term_labeling.token_label(tree.node_token(idx))
        lhs.add_arg([label])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_term(DCP_index(0, edge_label=tree.node_token(idx).edge()), [])])
        gram.add_rule(lhs, [], dcp=[dcp_rule])
        return lhs.nont()

    if not len(tree.children(idx)) <= 2:
        raise ValueError("Tree is not prebinarized!", tree, idx)

    children = [(child, join_spans(tree.fringe(child)))
                for child in tree.children(idx)]
    edge_labels = []
    for (low, high) in spans:
        arg = []
        pos = low
        while pos <= high:
            child_num = 0
            for i, (child, child_spans) in enumerate(children):
                for j, (child_low, child_high) in enumerate(child_spans):
                    if pos == child_low:
                        if tree.is_leaf(child) and not isolate_pos:
                            arg += [term_labeling.token_label(tree.node_token(child))]
                            edge_labels += [tree.node_token(child).edge()]
                        else:
                            arg += [LCFRS_var(child_num, j)]
                        pos = child_high + 1
                if not tree.is_leaf(child) or isolate_pos:
                    child_num += 1
        lhs.add_arg(arg)

    dcp_term_args = []
    rhs = []
    nont_counter = 0
    term_counter = 0
    for (child, child_spans) in children:
        if not tree.is_leaf(child) or isolate_pos:
            rhs_nont_fanout = len(child_spans)
            rhs += [nont_labeling.label_nont(tree, child) + '/' + str(rhs_nont_fanout)]
            dcp_term_args.append(DCP_var(nont_counter, 0))
            nont_counter += 1
        else:
            dcp_term_args.append(DCP_term(DCP_index(term_counter, edge_label=edge_labels[term_counter]), []))
            term_counter += 1

    dcp_lhs = DCP_var(-1, 0)

    label = tree.node_token(idx).category()
    if re.match(r'.*\|<.*>', label):
        dcp_term = dcp_term_args
    else:
        dcp_term = [DCP_term(DCP_string(label, edge_label=tree.node_token(idx).edge()), dcp_term_args)]
    dcp_rule = [DCP_rule(dcp_lhs, dcp_term)]

    gram.add_rule(lhs, rhs, dcp=dcp_rule)

    for (child, _) in children:
        if not tree.is_leaf(child) or isolate_pos:
            direct_extract_lcfrs_prebinarized_recur(tree, child, gram, term_labeling, nont_labeling, isolate_pos)
    return nont


def shift_var(elem):
    if isinstance(elem, LCFRS_var):
        return LCFRS_var(elem.mem - 1, elem.arg)
    else:
        return elem


def shift_dcp_vars(elems):
    counter = 0
    elems_shifted = []
    for elem in elems:
        if isinstance(elem, DCP_index):
            elems_shifted.append(DCP_term(DCP_index(counter, elem.edge_label()), arg=[]))
            counter += 1
        elif isinstance(elem, DCP_var):
            elems_shifted.append(elem)
        else:
            assert False
    return elems_shifted


def binarize(lhs, rhs, dcp_rule, hmarkov=0):
    if len(rhs) < 3:
        return [(lhs,rhs,dcp_rule)]

    rules = []
    args = copy.deepcopy(lhs.args())
    rhs_remain = copy.deepcopy(rhs)
    origin_counter = 0
    rule_head = dcp_rule[0].rhs()[0].head()
    dcp_args = [elem.head() for elem in dcp_rule[0].rhs()[0].arg() if isinstance(elem, DCP_term)]
    # print([str(elem) for elem in dcp_rule[0].rhs()[0].arg()])
    # print(dcp_rule[0], dcp_args)
    # dcp_vars = [DCP_var(0, 0), DCP_var(1, 0)]

    def strip_fanout(nont):
        return nont.split("/")[:-1]

    def bar_nont(fanout, nont_pos):
        rhs_context = []
        for prev_pos in range(nont_pos):
            if prev_pos >= nont_pos - hmarkov:
                rhs_context += strip_fanout(rhs[prev_pos])
        return "/".join(["BAR"] + strip_fanout(lhs.nont()) + rhs_context + [str(fanout)])

    while len(rhs_remain) > 2:
        lhs_args = []
        bar_args = []
        dcp_term_args = []
        bar_indices = []
        for arg in args:
            new_arg = []
            bar_arg = []
            tmp_arg = []
            tmp_dcp_indices = []
            for elem in arg:
                if isinstance(elem, LCFRS_var):
                    if elem.mem == 0:
                        if bar_arg:
                            new_arg.append(LCFRS_var(1, len(bar_args)))
                            if DCP_var(1, 0) not in dcp_term_args:
                                dcp_term_args.append(DCP_var(1, 0))
                            bar_args.append(bar_arg)
                            bar_arg = []
                        new_arg += tmp_arg
                        dcp_term_args += tmp_dcp_indices
                        tmp_arg = []
                        tmp_dcp_indices = []
                        new_arg.append(elem)
                        if DCP_var(0, 0) not in dcp_term_args:
                            dcp_term_args.append(DCP_var(0, 0))
                    else:
                        bar_arg += tmp_arg
                        bar_indices += tmp_dcp_indices
                        tmp_arg = []
                        tmp_dcp_indices = []
                        bar_arg.append(elem)
                else:
                    tmp_arg.append(elem)
                    tmp_dcp_indices.append(dcp_args[0])
                    dcp_args = dcp_args[1:]
            if bar_arg:
                new_arg.append(LCFRS_var(1, len(bar_args)))
                bar_args.append(bar_arg)
                if DCP_var(1, 0) not in dcp_term_args:
                    dcp_term_args.append(DCP_var(1, 0))
            new_arg += tmp_arg
            dcp_term_args += tmp_dcp_indices
            lhs_args.append(new_arg)
        bar_args = list(map(lambda xs: list(map(shift_var, xs)), bar_args))
        lhs_nont = lhs.nont() if origin_counter == 0 else bar_nont(len(lhs_args), origin_counter)
        lhs_new = LCFRS_lhs(lhs_nont)
        for arg in lhs_args:
            lhs_new.add_arg(arg)

        dcp_lhs = DCP_var(-1, 0)
        # dcp_indices = [DCP_term(DCP_index(i, idx.edge_label()), arg=[]) for i, idx in enumerate(dcp_indices)]
        dcp_term_args = shift_dcp_vars(dcp_term_args)
        if origin_counter == 0:
            dcp_term = [DCP_term(head=rule_head, arg=dcp_term_args)]
        else:
            dcp_term = dcp_term_args
        dcp_rule = DCP_rule(dcp_lhs, dcp_term)

        rules.append((lhs_new, rhs_remain[0:1] + [bar_nont(len(bar_args), origin_counter + 1)], [dcp_rule]))
        args = bar_args
        dcp_args = bar_indices
        rhs_remain = rhs_remain[1:]
        origin_counter += 1

    lhs_nont = lhs.nont() if origin_counter == 0 else bar_nont(len(args), origin_counter)
    lhs_new = LCFRS_lhs(lhs_nont)
    dcp_term_args = []
    term_counter = 0
    for arg in args:
        lhs_new.add_arg(arg)
        for elem in arg:
            if isinstance(elem, LCFRS_var):
                if DCP_var(elem.mem, 0) not in dcp_term_args:
                    dcp_term_args.append(DCP_var(elem.mem, 0))
            else:
                dcp_term_args.append(DCP_term(DCP_index(term_counter, dcp_args[term_counter].edge_label()), []))
                term_counter += 1

    dcp_lhs = DCP_var(-1, 0)
    # dcp_indices = [DCP_term(DCP_index(i, idx.edge_label()), []) for i, idx in enumerate(dcp_args)]

    if origin_counter == 0:
        dcp_term = [DCP_term(head=rule_head, arg=dcp_term_args)]
    else:
        dcp_term = dcp_term_args
    dcp_rule = DCP_rule(dcp_lhs, dcp_term)
    rules.append((lhs_new, rhs_remain, [dcp_rule]))
    return rules


############################################################
# Induction via unlabelled structure (recursive partitioning).


def fringe_extract_lcfrs(tree, fringes, naming='strict', term_labeling=PosTerminals(), isolate_pos=False, feature_logging=None):
    """
    :type tree: ConstituentTree
    :param fringes: recursive partitioning
    :param naming: 'strict' or 'child'
    :type naming: str
    :type term_labeling: ConstituentTerminalLabeling
    :rtype: LCFRS
    Get LCFRS for tree.
    """
    gram = LCFRS(start=START)
    first = None
    if len(tree.id_yield()) == 1 and isolate_pos:
        idx = tree.id_yield()[0]
        if tree.root[0] != idx:
            c_nont, c_spans, c_id_seq, c_nont_feat \
                = fringe_extract_lcfrs_recur(tree, fringes, gram, naming, term_labeling, isolate_pos, feature_logging,
                                             yield_one_check=False)

            fringe = fringes[0]
            spans = join_spans(fringe)
            args = []
            term_to_pos = {}  # maps input position to position in LCFRS rule
            for span in spans:
                args += [span_to_arg(span, [c_spans], tree, term_to_pos, term_labeling)]

            id_seq = make_id_seq(tree, tree.root[0], fringe)

            dcp_rules = []
            for (i, seq) in enumerate(id_seq):
                dcp_rhs = make_fringe_terms(tree, seq, [c_id_seq], term_to_pos, term_labeling)
                dcp_lhs = DCP_var(-1, i)
                dcp_rule = DCP_rule(dcp_lhs, dcp_rhs)
                dcp_rules += [dcp_rule]

            nont = id_nont(id_seq, tree, naming) + '/' + str(len(spans))
            nont_feat = feats(id_seq, tree)
            lhs = LCFRS_lhs(nont)
            for arg in args:
                lhs.add_arg(arg)
            rule = gram.add_rule(lhs, [c_nont], dcp=dcp_rules)
            if feature_logging is not None:
                feature_logging[(nont, nont_feat)] += 1
                feature_logging[(rule.get_idx(), nont_feat, tuple([c_nont_feat]))] += 1

            first = nont

    if first is None:
        (first, _, _, _) = fringe_extract_lcfrs_recur(tree, fringes, gram, naming, term_labeling, isolate_pos, feature_logging)
    lhs = LCFRS_lhs(START)
    lhs.add_arg([LCFRS_var(0, 0)])
    dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_var(0, 0)])
    gram.add_rule(lhs, [first], dcp=[dcp_rule])
    return gram


def fringe_extract_lcfrs_recur(tree, fringes, gram, naming, term_labeling, isolate_pos, feature_logging,
                               yield_one_check=True):
    """
    :type tree: ConstituentTree
    :param fringes: recursive partitioning
    :type gram: LCFRS
    :type naming: str
    :type term_labeling: ConstituentTerminalLabeling
    :rtype: (LCFRS, list[(int,int)], list[str])
    Traverse through recursive partitioning.
    """
    (fringe, children) = fringes
    nonts = []
    child_spans = []
    child_seqs = []
    child_feats = []
    for child in children:
        (child_nont, child_span, child_seq, child_feat) = \
            fringe_extract_lcfrs_recur(tree, child, gram, naming, term_labeling, isolate_pos, feature_logging)
        nonts += [child_nont]
        child_spans += [child_span]
        child_seqs += [child_seq]
        child_feats += [child_feat]
    spans = join_spans(fringe)
    term_to_pos = {}  # maps input position to position in LCFRS rule
    args = []
    for span in spans:
        args += [span_to_arg(span, child_spans, tree, term_to_pos, term_labeling)]
    # root[0] is legacy for single-rooted constituent trees
    id_seq = make_id_seq_single_pos(tree, tree.root[0], fringe, yield_one_check) if isolate_pos \
        else make_id_seq(tree, tree.root[0], fringe)
    dcp_rules = []
    for (i, seq) in enumerate(id_seq):
        dcp_rhs = make_fringe_terms(tree, seq, child_seqs, term_to_pos, term_labeling)
        dcp_lhs = DCP_var(-1, i)
        dcp_rule = DCP_rule(dcp_lhs, dcp_rhs)
        dcp_rules += [dcp_rule]
    nont = id_nont(id_seq, tree, naming) + '/' + str(len(spans))
    if feature_logging is not None:
        nont_feat = feats(id_seq, tree)
    else:
        nont_feat = None
    lhs = LCFRS_lhs(nont)
    for arg in args:
        lhs.add_arg(arg)
    rule = gram.add_rule(lhs, nonts, dcp=dcp_rules)
    if feature_logging is not None:
        feature_logging[(nont, nont_feat)] += 1
        feature_logging[(rule.get_idx(), nont_feat, tuple(child_feats))] += 1
    return nont, spans, id_seq, nont_feat


STRICT_MARKOV_REGEX = re.compile(r'strict-markov-v-(\d+)-h-(\d+)(-span)?')
STRICT_REGEX = re.compile(r'strict(-span)?')
CHILD_REGEX = re.compile(r'child(-span)?')


def compute_span_features(id_seq, tree):
    """
    :param id_seq:
    :type id_seq:
    :param tree:
    :type tree: ConstituentTree
    :return:
    :rtype: str
    """
    min_index = None  # len(tree.id_yield())
    max_index = None  # 0
    for ids in id_seq:
        first = min(tree.fringe(ids[0]))
        last = max(tree.fringe(ids[-1]))
        if min_index is None or min_index > first:
            min_index = first
        if max_index is None or max_index < last:
            max_index = last
    assert min_index is not None
    assert max_index is not None
    left_pos = tree.leaf_pos(tree.index_leaf(min_index))
    right_pos = tree.leaf_pos(tree.index_leaf(max_index))
    return '-span(' + left_pos + '-' + right_pos + ')'


def id_nont(id_seq, tree, naming):
    """
    :type id_seq: list[list[str]]
    :type tree: ConstituentTree
    :type naming: str
    :rtype: str
    Past labels of ids together.
    """
    sf = ""
    m = STRICT_MARKOV_REGEX.match(naming)
    if m:
        h = int(m.group(2))
        v = int(m.group(1))
        if m.group(3) == '-span':
            sf = compute_span_features(id_seq, tree)
        return id_nont_markov(id_seq, tree, v, h) + sf
    m = STRICT_REGEX.match(naming)
    if m:
        if m.group(1) == '-span':
            sf = compute_span_features(id_seq, tree)
        return id_nont_strict(id_seq, tree) + sf
    m = CHILD_REGEX.match(naming)
    if m:
        if m.group(1) == '-span':
            sf = compute_span_features(id_seq, tree)
        return id_nont_child(id_seq, tree) + sf

    raise Exception('unknown naming ' + naming)


def token_to_features(token, isleaf=True):
    id_feat = [("function", token.edge())]
    if isleaf:
        id_feat += [('form', token.form()), ('lemma', token.lemma()), ('pos', token.pos())]
        if isinstance(token.morph_feats(), str):
            assert(False and "Cannot handle morphological feature string!")
        else:
            assert(isinstance(token.morph_feats(), list))
            id_feat += token.morph_feats()
    else:
        id_feat += [('category', token.category())]
    return id_feat


def feats(id_seqs, tree):
    seqs_feats = []
    for id_seq in id_seqs:
        seq_feats = []
        for id in id_seq:
            token = tree.node_token(id)
            id_feat = token_to_features(token, isleaf=tree.is_leaf(id))
            seq_feats.append(frozenset(id_feat))
        seqs_feats.append(tuple(seq_feats))
    return tuple(seqs_feats)


def id_nont_strict(id_seqs, tree):
    """
    :type id_seqs: [[str]]
    :type tree: ConstituentTree
    :rtype: str
    Making naming on exact derived nonterminals.
    Consecutive children are separated by /.
    Where there is child missing, we have -.
    """
    s = ''
    for i, seq in enumerate(id_seqs):
        for j, id in enumerate(seq):
            if tree.is_leaf(id):
                s += tree.leaf_pos(id)
            else:
                s += tree.node_token(id).category()
            if j < len(seq) - 1:
                s += '/'
        if i < len(id_seqs) - 1:
            s += '-'
    return s


def id_nont_markov(id_seqs, tree, v=0, h=1, cutoff_symbol='...'):
    """
    :type id_seqs: [[str]]
    :type tree: ConstituentTree
    :rtype: str
    Making naming on exact derived nonterminals
    while markovizing sequences of consecutive children.
    Consecutive children are separated by /.
    Where there is child missing, we have -.
    """
    ss = []
    for i, seq in enumerate(id_seqs):
        s = []
        p = []
        _v = 0
        pid = seq[0]
        while _v < v:
            pid = tree.parent(pid)
            if pid is None:
                break
            p.append(tree.node_token(pid).category())
            _v += 1
        for j, idx in enumerate(seq):
            if j < h:
                if tree.is_leaf(idx):
                    s.append(tree.leaf_pos(idx))
                else:
                    s.append(tree.node_token(idx).category())
            else:
                s.append(cutoff_symbol)
                break
        if p:
            ss.append('|'.join(p) + '|' + '/'.join(s))
        else:
            ss.append('/'.join(s))
    return '-'.join(ss)


def id_nont_child(id_seq, tree):
    """
    :type id_seq: list[list[str]]
    :type tree: ConstituentTree
    :rtype: str
    Replace consecutive siblings by mention of parent.
    """
    s = ''
    for i, seq in enumerate(id_seq):
        if len(seq) == 1:
            if tree.is_leaf(seq[0]):
                s += tree.leaf_pos(seq[0])
            else:
                s += tree.node_token(seq[0]).category()
        else:
            id = tree.parent(seq[0])
            s += 'children_of_' + tree.node_token(id).category()
        if i < len(id_seq) - 1:
            s += '-'
    return s


def make_id_seq(tree, id, fringe):
    """
    :type tree: ConstituentTree
    :type id: str
    :type fringe: list[int]
    Compute list of lists of adjacent nodes whose fringes
    are included in input fringe.
    """
    if set(tree.fringe(id)) - fringe == set():
        # fully included in fringe
        return [[id]]
    else:
        seqs = []
        seq = []
        for child in tree.children(id):
            child_fringe = set(tree.fringe(child))
            if child_fringe - fringe == set():
                # fully included in fringe
                seq += [child]
            elif child_fringe & fringe != set():
                # overlaps with fringe
                if seq:
                    seqs += [seq]
                    seq = []
                seqs += make_id_seq(tree, child, fringe)
            elif seq:
                # not included in fringe, and breaks sequence
                seqs += [seq]
                seq = []
        if seq:
            seqs += [seq]
        return seqs


def make_id_seq_single_pos(tree, id, fringe, yield_one_check=True):
    """
    :type tree: ConstituentTree
    """
    if yield_one_check and len(tree.id_yield()) == 1:
        assert set(tree.fringe(id)) - fringe == set()
        return [[id]]
    else:
        if len(fringe) == 1:
            position = list(fringe)[0]
            return [[tree.id_yield()[position]]]
        else:
            return make_id_seq(tree, id, fringe)


def make_fringe_terms(tree, seq, child_seqss, term_to_pos, term_labeling):
    """
    :type tree: ConstituentTree
    :type seq: list[str]
    :type child_seqss: list[list[int]]
    :param term_to_pos: maps int to int (input position to position in LCFRS rule)
    :type term_labeling: ConstituentTerminalLabeling
    :return: list of DCP_term/DCP_index/DCP_var
    Make expression in terms of variables for the RHS members.
    Repeatedly replace nodes by variables.
    """
    for i, child_seqs in enumerate(child_seqss):
        for j, child_seq in enumerate(child_seqs):
            k = sublist_index(child_seq, seq)
            if k >= 0:
                var = DCP_var(i, j)
                seq = seq[:k] + [var] + seq[k + len(child_seq):]
    terms = []
    for elem in seq:
        if isinstance(elem, DCP_var):
            terms += [elem]
        else:
            if tree.is_leaf(elem):
                k = tree.leaf_index(elem)
                pos = term_to_pos[k]
                terms.append(DCP_term(DCP_index(pos, tree.node_token(elem).edge(), tree.node_token(elem).pos()), []))
            else:
                lab = tree.node_token(elem).category()
                arg = make_fringe_terms(tree, tree.children(elem), \
                                        child_seqss, term_to_pos, term_labeling)
                string = DCP_string(lab)
                string.set_edge_label(tree.node_token(elem).edge())
                terms.append(DCP_term(string, arg))
    return terms


def span_to_arg(span, children, tree, term_to_pos, term_labeling):
    """
    :type low: int
    :type high: int
    :type children: list[(int, int)]
    :type tree: ConstituentTree
    :param term_to_pos: maps int to int (input position to position in LCFRS rule)
    :type term_labeling: ConstituentTerminalLabeling
    :return: list of LCFRS_var/string
    Turn span into LCFRS arg (variables and terminals).
    Children is list of (sub)spans.
    Also keep list of terminals.
    """
    low, high = span
    arg = []
    k = low
    while k <= high:
        match = False
        for i, child in enumerate(children):
            for j, (child_low, child_high) in enumerate(child):
                if child_low == k:
                    arg += [LCFRS_var(i, j)]
                    k = child_high + 1
                    match = True
        if not match:
            arg += [term_labeling.token_label(tree.token_yield()[k], _loc=k)]
            term_to_pos[k] = len(term_to_pos.keys())
            k += 1
    return arg


#####################################################
# Auxiliary.
def sublist_index(s, l):
    """
    :type s: list[int]
    :type l: list[str]
    :rtype: int
    In list of strings and variables, find index of sublist.
    Return -1 if none.
    """
    for k in range(len(l) - len(s) + 1):
        if s == l[k:k + len(s)]:
            return k
    return -1


__all__ = [direct_extract_lcfrs,
           direct_extract_lcfrs_from_prebinarized_corpus,
           fringe_extract_lcfrs,
           BasicNonterminalLabeling,
           NonterminalsWithFunctions]