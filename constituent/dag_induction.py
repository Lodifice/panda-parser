import re

from grammar.lcfrs import LCFRS, LCFRS_lhs, LCFRS_var
from grammar.dcp import DCP_rule, DCP_var, DCP_term, DCP_index, DCP_string
from grammar.induction.terminal_labeling import PosTerminals
from constituent.induction import BasicNonterminalLabeling
from grammar.induction.decomposition import join_spans
from hybridtree.general_hybrid_tree import HybridDag

START = "START"
SECEDGE = "SECEDGE"


def direct_extract_lcfrs_from_prebinarized_corpus(tree,
                                                  term_labeling=PosTerminals(),
                                                  nont_labeling=BasicNonterminalLabeling(),
                                                  isolate_pos=True):
    gram = LCFRS(start=START)
    root = tree.root[0]
    if root in tree.full_yield():
        lhs = LCFRS_lhs(START)
        label = term_labeling.token_label(tree.node_token(root))
        lhs.add_arg([label])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_term(DCP_index(0, edge_label=tree.node_token(root).edge()), [])])
        gram.add_rule(lhs, [], dcp=[dcp_rule])
    else:
        first, _, _ = direct_extract_lcfrs_prebinarized_recur(tree, root, gram, term_labeling, nont_labeling, isolate_pos)
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
    assert isinstance(tree, HybridDag)
    fringe = tree.fringe(idx)
    spans = join_spans(fringe)
    nont_fanout = len(spans)

    _bot = list(bottom(tree, [idx] + tree.descendants(idx)))
    _top = list(top(tree, [idx] + tree.descendants(idx)))

    nont = nont_labeling.label_nont(tree, idx) + '/' + '/'.join(map(str, [nont_fanout, len(_bot), len(_top)]))

    lhs = LCFRS_lhs(nont)

    if idx in tree.full_yield():
        label = term_labeling.token_label(tree.node_token(idx))
        lhs.add_arg([label])
        dcp_rule = DCP_rule(DCP_var(-1, 0), [DCP_term(DCP_index(0, edge_label=tree.node_token(idx).edge()), [])])
        gram.add_rule(lhs, [], dcp=[dcp_rule])
        return lhs.nont(), _bot, _top

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
                        if child in tree.full_yield() and not isolate_pos:
                            arg += [term_labeling.token_label(tree.node_token(child))]
                            edge_labels += [tree.node_token(child).edge()]
                        else:
                            arg += [LCFRS_var(child_num, j)]
                        pos = child_high + 1
                if child not in tree.full_yield() or isolate_pos:
                    child_num += 1
        lhs.add_arg(arg)

    dcp_term_args = []
    rhs = []
    nont_counter = 0
    term_counter = 0

    cbots = []
    ctops = []

    for (child, child_spans) in children:

        if child not in tree.full_yield() or isolate_pos:
            c_nont, _cbot, _ctop = direct_extract_lcfrs_prebinarized_recur(tree, child, gram, term_labeling, nont_labeling, isolate_pos)
            rhs.append(c_nont)
            cbots.append(_cbot)
            ctops.append(_ctop)
            dcp_term_args.append(DCP_var(nont_counter, len(_cbot) + _ctop.index(child)))
            nont_counter += 1
        else:
            dcp_term_args.append(DCP_term(DCP_index(term_counter, edge_label=edge_labels[term_counter]), []))
            term_counter += 1

    for sec, sec_child in enumerate(tree.sec_children(idx)):
        if sec_child not in tree.descendants(idx):
            print(idx, "has external", sec_child)
            assert sec_child in _bot
            dcp_term_args.append(DCP_term(DCP_string(SECEDGE, edge_label=tree.sec_child_edge_labels(idx)[sec]),
                                          [DCP_var(-1, _bot.index(sec_child))]))

        else:
            print(idx, "has internal", sec_child)

            assert False

    dcp_lhs = DCP_var(-1, len(_bot) + _top.index(idx))

    label = tree.node_token(idx).category()
    if re.match(r'.*\|<.*>', label):
        dcp_term = dcp_term_args
    else:
        dcp_term = [DCP_term(DCP_string(label, edge_label=tree.node_token(idx).edge()), dcp_term_args)]
    dcp_rule = DCP_rule(dcp_lhs, dcp_term)

    dcp_rules = [dcp_rule]

    for top_idx in _top:
        if top_idx != idx:
            # must be in some child
            rule = None

            for nont_counter, _ctop in enumerate(ctops):
                if top_idx in _ctop:
                    rule = DCP_rule(DCP_var(-1, len(_bot) + _top.index(top_idx)),
                                    [DCP_var(nont_counter, len(cbots[nont_counter]) + _ctop.index(top_idx))])

                    break
            assert rule is not None
            dcp_rules.append(rule)

    for nont_counter, _cbot in enumerate(cbots):
        for bot_idx in _cbot:
            rule = None
            rule_lhs = DCP_var(nont_counter, _cbot.index(bot_idx))

            if bot_idx in _bot:
                rule = DCP_rule(rule_lhs, [DCP_var(-1, _bot.index(bot_idx))])
            else:
                for nont_counter2, _ctop in enumerate(ctops):
                    if bot_idx in _ctop:
                        rule = DCP_rule(rule_lhs,
                                        [DCP_var(nont_counter2, len(cbots[nont_counter2]) + _ctop.index(bot_idx))])
                        break
            assert rule is not None
            dcp_rules.append(rule)

    gram.add_rule(lhs, rhs, dcp=dcp_rules)

    return nont, _bot, _top


def top(tree, idxs):
    """
    :param tree:
    :type tree: HybridDag
    :param idxs:
    :type idxs:
    :return:
    :rtype:
    """
    top_nodes = {idx for idx in idxs
                 if idx in tree.root
                 or tree.parent(idx) not in idxs
                 or not all([sec_parent in idxs for sec_parent in tree.sec_parents(idx)])
                 }
    return top_nodes


def bottom(tree, idxs):
    """
    :param tree:
    :type tree: HybridDag
    :param idxs:
    :type idxs:
    :return:
    :rtype:
    """
    bottom_nodes = {child for idx in idxs for child in tree.sec_children(idx) if child not in idxs}
    return bottom_nodes


__all__ = [direct_extract_lcfrs_from_prebinarized_corpus]
