from __future__ import print_function
import random
from libcpp.vector cimport vector
from dependency.top_bottom_max import top_max, bottom_max
from collections import OrderedDict

# Auxiliary routines for dealing with spans in an input string.

# ################################################################


# For list of indices, order and join into contiguous sequences.
# A sequence is represented by (low, high), where high is last
# element.
# indices: list of int
# return: list of pair of int
cpdef list join_spans(vector[int] indices):
    indices = sorted(set(indices))
    spans = []
    cdef int low = -1
    cdef int i
    cdef int high
    for i in indices:
        if low < 0:
            low = i
            high = i
        elif i == high + 1:
            high = i
        else:
            spans += [(low, high)]
            low = i
            high = i
    if low >= 0:
        spans += [(low, high)]
    return spans


# For a list of spans, replace by indices.
# spans: list of pair of int
# return: list of int
def expand_spans(spans):
    return sorted(set([i for span in spans
                       for i in range(span[0], span[1] + 1)]))


###############################################################

# Recursive partitioning of input string into substrings.
# A recursive partitioning is represented as a pair of ints
# and a list of recursive partitionings.


def left_branching_partitioning(int len):
    """
    :param len: length of sentence
    :type len: int
    :return: left-branching recursive partitioning of length **len**
    """
    if len == 0:
        return set(), []
    elif len == 1:
        return {0}, []
    else:
        return (set(range(len)), [
            left_branching_partitioning(len - 1),
            ({len - 1}, [])])


def right_branching_partitioning(int len):
    """
    :param len: length of sentence
    :type len: int
    :return: right-branching recursive partitioning of length **len**
    """
    return right_branching_partitioning_recur(0, len)


def right_branching_partitioning_recur(int low, int high):
    if low >= high:
        return set(), []
    elif low == high - 1:
        return {low}, []
    else:
        return (set(range(low, high)), [
            ({low}, []),
            right_branching_partitioning_recur(low + 1, high)])


def fanout_limited_partitioning(part, int fanout):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :return: binarized recursive partitioning with fanout <= **fanout**
    Transform existing partitioning to limit number of spans.
    Breadth-first search among descendants (from right to left) for subpartitioning that stays within fanout.
    """

    (root, children) = part
    agenda = children[::-1]  # reversed to favour left branching
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                child2 = restrict_part([(rest, children)], rest)[0]
                child1_restrict = fanout_limited_partitioning(child1, fanout)
                child2_restrict = fanout_limited_partitioning(child2, fanout)
                return root, sort_part(child1_restrict, child2_restrict)
            else:
                next_agenda += subchildren[::-1]  # reversed
        agenda = next_agenda
    return part


def fanout_limited_partitioning_left_to_right(part, int fanout):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :return: binarized recursive partitioning with fanout <= **fanout**
    Transform existing partitioning to limit number of spans.
    Breadth-first search among descendants (left to right) for subpartitioning that stays within fanout.
    """
    (root, children) = part
    agenda = children
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                child2 = restrict_part([(rest, children)], rest)[0]
                child1_restrict = fanout_limited_partitioning_left_to_right(child1, fanout)
                child2_restrict = fanout_limited_partitioning_left_to_right(child2, fanout)
                return root, sort_part(child1_restrict, child2_restrict)
            else:
                next_agenda += subchildren
        agenda = next_agenda
    return part


def fanout_limited_partitioning_argmax(part, int fanout):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :return: binarized recursive partitioning with fanout <= **fanout**
    Transform existing partitioning to limit number of spans.
    Choose position p such that p = argmax |part(p)|
    """
    (root, children) = part
    if children == []:
        return part
    agenda = children
    argmax = None
    argroot = {}
    argchildren = []
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                    if argmax is None or len(subroot) > len(argroot):
                        argmax = child1
                        (argroot, argchildren) = argmax
            else:
                next_agenda += subchildren
        agenda = next_agenda
    rest = remove_spans_from_spans(root, argroot)
    child2 = restrict_part([(rest, children)], rest)[0]
    child1_restrict = fanout_limited_partitioning_argmax(argmax, fanout)
    child2_restrict = fanout_limited_partitioning_argmax(child2, fanout)
    return root, sort_part(child1_restrict, child2_restrict)


def fanout_limited_partitioning_no_new_nont(part, fanout, tree, nonts, nont_labelling, fallback):
    """
    :param part: recursive partitioning
    :type fanout: int
    :type tree: HybridTree
    :param nonts: list/set of nonterminals
    :type nont_labelling: AbstractLabeling
    :param fallback: fallback strategy if no node can safely be choosen [-rtl,-ltr,-argmax,-random]
    :type fallback: str
    :return: binarized recursive partitioning with fanout <= **fanout**
    Transform existing partitioning to limit number of spans.
    Choose position p such that no new nonterminal is added to grammar if possible.
    """
    (root, children) = part
    agenda = children

    oneIn = None
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                # check if first nonterminal was already created
                subindex = []
                for pos in subroot:
                    subindex += [tree.index_node(pos+1)]
                positions = map(int, subroot)
                b_max = bottom_max(tree, subindex)
                t_max = top_max(tree, subindex)
                spans = join_spans(positions)
                nont = nont_labelling.label_nonterminal(tree, subindex, t_max, b_max, len(spans))
                if nont in nonts:
                    child2 = restrict_part([(rest, children)], rest)[0]
                    (subroot2, subchildren2) = child2
                    # check if second nonterminal was already created
                    subindex2 = []
                    for pos in subroot2:
                        subindex2 += [tree.index_node(pos+1)]
                    positions2 = map(int, subroot2)
                    b_max2 = bottom_max(tree, subindex2)
                    t_max2 = top_max(tree, subindex2)
                    spans2 = join_spans(positions2)
                    nont2 = nont_labelling.label_nonterminal(tree, subindex2, t_max2, b_max2, len(spans2))

                    if nont2 in nonts:
                        child1_restrict = fanout_limited_partitioning_no_new_nont(child1, fanout, tree, nonts, nont_labelling, fallback)
                        child2_restrict = fanout_limited_partitioning_no_new_nont(child2, fanout, tree, nonts, nont_labelling, fallback)
                        return root, sort_part(child1_restrict, child2_restrict)
                    elif oneIn is None:
                        oneIn = (child1, child2)

            next_agenda += subchildren
        agenda = next_agenda

    # check if at least one candidate was found:
    if oneIn is not None:
        (child1, child2) = oneIn
        child1_restrict = fanout_limited_partitioning_no_new_nont(child1, fanout, tree, nonts, nont_labelling, fallback)
        child2_restrict = fanout_limited_partitioning_no_new_nont(child2, fanout, tree, nonts, nont_labelling, fallback)
        return root, sort_part(child1_restrict, child2_restrict)

    if fallback == '-rtl':
        return fallback_rtl(part, fanout, tree, nonts, nont_labelling, fallback)
    elif fallback == '-ltr':
        return fallback_ltr(part, fanout, tree, nonts, nont_labelling, fallback)
    elif fallback == '-argmax':
        return fallback_argmax(part, fanout, tree, nonts, nont_labelling, fallback)
    else:
        return fallback_random(part, fanout, tree, nonts, nont_labelling, fallback)


def fallback_random(part, fanout, tree, nonts, nont_labelling, fallback):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :type tree: HybridTree
    :param nonts:  list/set of nonterminals
    :type nont_labelling: AbstractLabeling
    :return: binarized recursive partitioning with fanout <= **fanout**
    Fallback function if fanout_limited_partitioning_no_new_nont_rec has
    not found a position corresponding to an existing nonterminal.
    """
    (root, children) = part
    agenda = children
    possibleChoices = []
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                possibleChoices += [child1]
            next_agenda += subchildren
        agenda = next_agenda
    if possibleChoices == []:
        return part
    chosen = random.choice(possibleChoices)
    chosen = tuple(chosen)
    (subroot, subchildren) = chosen
    rest = remove_spans_from_spans(root, subroot)
    child2 = restrict_part([(rest, children)], rest)[0]
    child1_restrict = fanout_limited_partitioning_no_new_nont(chosen, fanout, tree, nonts, nont_labelling, fallback)
    child2_restrict = fanout_limited_partitioning_no_new_nont(child2, fanout, tree, nonts, nont_labelling, fallback)
    return root, sort_part(child1_restrict, child2_restrict)


def fallback_argmax(part, fanout, tree, nonts, nont_labelling, fallback):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :type tree: HybridTree
    :param nonts:  list/set of nonterminals
    :type nont_labelling: AbstractLabeling
    :return: binarized recursive partitioning with fanout <= **fanout**
    Fallback function if fanout_limited_partitioning_no_new_nont_rec has
    not found a position corresponding to an existing nonterminal.
    """
    (root, children) = part
    if children == []:
        return part
    agenda = children
    argmax = None
    argroot = {}
    argchildren = []
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                    if argmax is None or len(subroot) > len(argroot):
                        argmax = child1
                        (argroot, argchildren) = argmax
            else:
                next_agenda += subchildren
        agenda = next_agenda
    rest = remove_spans_from_spans(root, argroot)
    child2 = restrict_part([(rest, children)], rest)[0]
    child1_restrict = fanout_limited_partitioning_no_new_nont(argmax, fanout, tree, nonts, nont_labelling, fallback)
    child2_restrict = fanout_limited_partitioning_no_new_nont(child2, fanout, tree, nonts, nont_labelling, fallback)
    return root, sort_part(child1_restrict, child2_restrict)


def fallback_ltr(part, fanout, tree, nonts, nont_labelling, fallback):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :type tree: HybridTree
    :param nonts:  list/set of nonterminals
    :type nont_labelling: AbstractLabeling
    :return: binarized recursive partitioning with fanout <= **fanout**
    Fallback function if fanout_limited_partitioning_no_new_nont_rec has
    not found a position corresponding to an existing nonterminal.
    """
    (root, children) = part
    agenda = children
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                child2 = restrict_part([(rest, children)], rest)[0]
                child1_restrict = fanout_limited_partitioning_no_new_nont(child1, fanout, tree, nonts, nont_labelling, fallback)
                child2_restrict = fanout_limited_partitioning_no_new_nont(child2, fanout, tree, nonts, nont_labelling, fallback)
                return root, sort_part(child1_restrict, child2_restrict)
            else:
                next_agenda += subchildren
        agenda = next_agenda
    return part


def fallback_rtl(part, fanout, tree, nonts, nont_labelling, fallback):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :type tree: HybridTree
    :param nonts:  list/set of nonterminals
    :type nont_labelling: AbstractLabeling
    :return: binarized recursive partitioning with fanout <= **fanout**
    Fallback function if fanout_limited_partitioning_no_new_nont_rec has
    not found a position corresponding to an existing nonterminal.
    """
    (root, children) = part
    agenda = children[::-1]  # reversed to favour left branching
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                child2 = restrict_part([(rest, children)], rest)[0]
                child1_restrict = fanout_limited_partitioning_no_new_nont(child1, fanout, tree, nonts, nont_labelling, fallback)
                child2_restrict = fanout_limited_partitioning_no_new_nont(child2, fanout, tree, nonts, nont_labelling, fallback)
                return root, sort_part(child1_restrict, child2_restrict)
            else:
                next_agenda += subchildren[::-1]  # reversed
        agenda = next_agenda
    return part


def fanout_limited_partitioning_random_choice(part, fanout):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :return: binarized recursive partitioning with fanout <= **fanout**
    Transform existing partitioning to limit number of spans.
    Choose subpartitioning that stays within fanout randomly.
    """
    (root, children) = part
    agenda = children
    possibleChoices = []
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                possibleChoices += [child1]
            next_agenda += subchildren
        agenda = next_agenda
    if possibleChoices == []:
        return part
    chosen = random.choice(possibleChoices)
    chosen = tuple(chosen)
    (subroot, subchildren) = chosen
    rest = remove_spans_from_spans(root, subroot)
    child2 = restrict_part([(rest, children)], rest)[0]
    child1_restrict = fanout_limited_partitioning_random_choice(chosen, fanout)
    child2_restrict = fanout_limited_partitioning_random_choice(child2, fanout)
    return root, sort_part(child1_restrict, child2_restrict)


def remove_spans_from_spans(spans1, spans2):
    """

    :param spans1: list of sentence positions
    :type spans1: list[int]
    :param spans2: list of sentence positions
    :type spans2: list[int]
    :rtype: set(int)
    Remove sentence positions in spans2 from spans1.
    """
    set1 = set(spans1)
    set2 = set(spans2)
    return set1 - set2


def restrict_part(part, relevant):
    """

    :param part: recursive partitioning
    :param relevant: set or list of sentence positions
    :type relevant:
    :return: list of recursive partitioning
    :rtype: list
    Remove from each subtree of **part** to **relevant**. Remove empty subtrees and collapse unary chains.
    """
    part_restrict = []
    for (root, children) in part:
        root_restrict = root & relevant
        if root_restrict != set():
            children_restrict = restrict_part(children, relevant)
            if len(children_restrict) == 1 and children_restrict[0][0] == root_restrict:
                part_restrict += [children_restrict[0]]
            else:
                part_restrict += [(root_restrict, children_restrict)]
    return part_restrict


cpdef int n_spans(vector[int] l):
    """
    :param l: list of sentence positions
    :type l: list[int]
    :return: minimum number of continuous intervals required to partition **l**
    :rtype: int
    """
    return len(join_spans(l))


def sort_part(part1, part2):
    """
    :param part1: recursive partitioning
    :param part2: recursive partitioning
    :return: list of two recursive partitionings
    For two disjoint partitionings, determine which one comes first. This is determined by first position.
    """
    (root1, _) = part1
    (root2, _) = part2
    if sorted(root1)[0] < sorted(root2)[0]:
        return [part1, part2]
    else:
        return [part2, part1]


def print_partitioning(part, level=0):
    """
    :param part: recursive partitioning
    :param level: level of indentation
    :type level: int
    """
    (root, children) = part
    print(' ' * level, root)
    for child in children:
        print_partitioning(child, level + 1)


cdef set compute_highest_nodes(set nodes, ref_bin):
    working_set = set(nodes)
    highest_nodes = set(nodes)
    cdef bint changed = True
    while changed:
        changed = False
        add_later = set()
        for node in working_set:
            if ref_bin.parent(node):
                parent = ref_bin.parent(node)
                if parent not in working_set \
                        and parent not in add_later \
                        and all([child in working_set for child in ref_bin.children(parent)]):
                    changed = True
                    add_later.add(parent)
                    for child in ref_bin.children(parent):
                        highest_nodes.remove(child)
                    highest_nodes.add(parent)
        working_set = working_set.union(add_later)
    return highest_nodes


cdef int depth(node, hybrid_tree):
    """
    :param node: node of **hybrid_tree**
    :type hybrid_tree: HybridTree
    :return: length of path from **node** to root of **hybrid_tree**
    :rtype: int
    """
    if node in hybrid_tree.root:
        return 0
    parent = hybrid_tree.parent(node)
    assert parent
    return 1 + depth(parent, hybrid_tree)


cpdef compute_candidates(list agenda, root, int fanout):
    """
    :param agenda: list of partitionings
    :type agenda: list
    :param root: set of sentence positions
    :type root: set[int]
    :param fanout: maximum fanout
    :type fanout: int
    :rtype: OrderedDict
    Compute set of nodes *p* such that *fanout(p) <=* **fanout**
    and *fanout(* **root** *\\ p)* <= **fanout**
    """
    candidates = OrderedDict()
    while agenda:
        sub_root, sub_children = agenda[0]
        rest = remove_spans_from_spans(root, sub_root)
        if n_spans(sub_root) <= fanout and n_spans(rest) <= fanout:
            candidates[frozenset(sub_root)] = sub_children
        agenda = agenda[1:] + sub_children
    return candidates

cpdef fanout_limit_partitioning_with_guided_binarization(part, int fanout, ref_bin):
    """
    :param part: recursive partitioning
    :param fanout: maximum fanout
    :type fanout: int
    :type ref_bin: HybridTree
    Transform existing partitioning to limit number of spans.
    Choose position p such that the subtree of **ref_bin** to which p corresponds
    is rooted as high as possible (resolve nondeterminism by choosing p breadth-first left-to-right.)
    """
    root, children = part

    if len(root) <= 1:
        return part

    candidates = compute_candidates(list(children), root, fanout)

    min_candidate = None
    cdef int min_depth = 9999999

    assert candidates

    cdef int high_depth
    for candidate in candidates:
        # find corresponding node in ref_bin
        nodes = {ref_bin.index_node(idx + 1) for idx in candidate}
        highest_nodes = compute_highest_nodes(nodes, ref_bin)
        high_depth = min([depth(node, ref_bin) for node in highest_nodes])

        # print(candidate, nodes, highest_nodes, high_depth)

        if high_depth < min_depth:
            min_depth = high_depth
            min_candidate = candidate

    # print(min_candidate, min_depth)
    assert min_candidate
    min_rest = remove_spans_from_spans(root, min_candidate)
    child2 = restrict_part([(min_rest, children)], min_rest)[0]
    child1_restrict = fanout_limit_partitioning_with_guided_binarization(
        (min_candidate, candidates[min_candidate]), fanout, ref_bin)
    child2_restrict = fanout_limit_partitioning_with_guided_binarization(child2, fanout, ref_bin)
    return root, sort_part(child1_restrict, child2_restrict)
