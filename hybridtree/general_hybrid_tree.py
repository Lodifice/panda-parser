#-*- coding: utf-8 -*-
__author__ = 'kilian'

from collections import defaultdict

from grammar.induction.decomposition import join_spans
from hybridtree.monadic_tokens import MonadicToken


class HybridTree:
    """
    A directed acyclic graph, where a (not necessarily strict) subset of the nodes is linearly ordered.
    """
    @property
    def virtual_root(self):
        return 'VROOT'

    def __init__(self, sent_label=None):
        """
        :param sent_label: name of the sentence
        :type sent_label: str
        """
        # label of sentence (identifier in corpus)
        self.__sent_label = sent_label
        # maps node id to list of ids of children
        self._id_to_child_ids = {self.virtual_root: []}
        # maps node id to token
        self._id_to_token = {}
        # maps node id to part-of-speech tag
        # self.__id_to_pos = {}
        # list of node ids in ascending order
        self.__ordered_ids = []
        # list of node ids in ascending order, including disconnected nodes
        self.__full_yield = []
        self._parent = {}
        # maps node id to position in the ordering
        # self.__id_to_node_index = {}
        # maps node_index (position in ordering) to node id
        # self.__node_index_to_id = {}
        # number of nodes in ordering
        # self.__n_ordered_nodes = 0
        # store dependency labels (DEPREL in ConLL)
        # self.__id_to_dep_label = {}

    def sent_label(self):
        """
        :rtype: str
        :return: name of the sentence
        """
        return self.__sent_label

    def add_to_root(self, id):
        """
        Set root to node given by id.
        :param id: node id
        :type id: str
        """
        self.add_child(self.virtual_root, id)

    @property
    def root(self):
        """
        :rtype: list of str
        :return: Id of root.
        """
        return self.children(self.virtual_root)

    def add_node(self, id, token, order=False, connected=True):
        """
        :param id: node id
        :type id: str
        :param token: word + pos, syntactic category
        :type token: MonadicToken
        :param order: include node in linear ordering
        :type order: bool
        :param connected: should the node be connected to other nodes
        :type connected: bool
        Add next node. Order of adding nodes is significant for ordering.
        Set order = True and connected = False to include some token (e.g. punctuation)
        that appears in the yield but shall be ignored during tree operations.
        """
        self._id_to_token[id] = token
        if order is True:
            if connected is True:
                self.__ordered_ids += [id]
                # self.__id_to_node_index[id] = self.__n_ordered_nodes
                # self.__n_ordered_nodes += 1
                # self.__node_index_to_id[self.__n_ordered_nodes] = id
            self.__full_yield += [id]

    def add_child(self, parent, child):
        """
        :param parent: id of parent node
        :type parent: str
        :param child: id of child node
        :type child: str
        Add a pair of node ids in the tree's parent-child relation.
        """
        if parent not in self._id_to_child_ids:
            self._id_to_child_ids[parent] = [child]
        else:
            self._id_to_child_ids[parent] += [child]
        self._parent[child] = parent

    def parent(self, id):
        """
        :rtype: str
        :param id: node id
        :type id: str
        :return: id of parent node, or None.
        """
        parent = self._parent.get(id, None) #__parent_recur(id, self.virtual_root)
        if parent == self.virtual_root:
            return None
        else:
            return parent

    def __parent_recur(self, child, id):
        """
        :rtype: str
        :param child: str (the node, whose parent is searched)
        :param id: (potential parent)
        :return:  id of parent node, or None.
        """
        if child in self.children(id):
            return id
        else:
            for next_id in self.children(id):
                parent = self.__parent_recur(child, next_id)
                if parent is not None:
                    return parent
        return None

    def reentrant(self):
        """
        :rtype: bool
        :return: Is there node that is child of two nodes?
        """
        parent = defaultdict(list)
        for id in self._id_to_child_ids:
            for child in self.children(id):
                parent[child] += [id]
        for id in parent:
            if len(parent[id]) > 1:
                return True
        return False

    def children(self, id):
        """
        :rtype: list[str]
        :param id: str
        :return: Get the list of node ids of child nodes, or the empty list.
        """
        if isinstance(id, list):
            pass
        if id in self._id_to_child_ids:
            return self._id_to_child_ids[id]
        else:
            return []

    def descendants(self, id):
        """
        :param id: node id
        :type id: str
        :return: the list of node ids of all "transitive" children
        :rtype: list[str]
        """
        des = []
        if id in self._id_to_child_ids:
            for id2 in self._id_to_child_ids[id]:
                des.append(id2)
                des += self.descendants(id2)
        return des

    def in_ordering(self, id):
        """
        :param id: node id
        :type id: str
        :return: Is the node in the ordering?
        :rtype: bool
        """
        return id in self.__ordered_ids

    def disconnected(self, id):
        """
        :param id: node id
        :type id: str
        :return: Is the node in the yield, but not connected to the root?
        :rtype: bool
        """
        return id in self.__full_yield and id not in self.__ordered_ids

    def index_node(self, index):
        """
        :param index: index in ordering
        :type index: int
        :return: node id at index in ordering
        :rtype: str
        """
        return self.__ordered_ids[index - 1]
        # return self.__node_index_to_id[index]

    def node_index(self, id):
        """
        :param id: node id
        :type id: str
        :return: index of node in ordering
        :rtype: int
        """
        return self.__ordered_ids.index(id)
        # return self.__id_to_node_index[id]

    def node_index_full(self, id):
        """
        :param id: node id
        :type id: str
        :return: index of node in full_yield
        :rtype: int
        """
        return self.__full_yield.index(id)

    def reorder(self):
        """
        Reorder children according to smallest node (w.r.t. ordering) in subtree.
        """
        self.__reorder(self.virtual_root)

    def __reorder(self, id):
        """
        :param id: node id
        :type id: str
        :return: index of smallest node in sub tree (or -1 if none exists)
        :rtype: int
        """
        min_indices = {}
        if self.children(id).__len__() > 0:
            for child in self.children(id):
                min_indices[child] = self.__reorder(child)
            self._id_to_child_ids[id] = sorted(self.children(id), key=lambda i: min_indices[i])
        if self.in_ordering(id):
            min_indices[id] = self.node_index(id)
        min_index = -1
        for index in min_indices.values():
            if min_index < 0 or index < min_index:
                min_index = index
        return min_index

    def fringe(self, id):
        """
        :param id: node id
        :type id: str
        :return: indices (w.r.t. ordering) of all nodes under some node, cf. \Pi^{-1} in paper
        :rtype: list[int]
        List of indices (w.r.t. ordering) obtained by pre-order traversal over the subtree starting at id.
        """
        y = []
        if self.in_ordering(id):
            y = [self.node_index(id)]
        for child in self.children(id):
            y += self.fringe(child)
        return y

    def n_spans(self, id):
        """
        :param id: node id
        :type id: str
        :return: Number of contiguous spans of node.
        :rtype : int
        """
        return len(join_spans(self.fringe(id)))

    def max_n_spans(self):
        """
        :return: Maximum number of spans of any node.
        :rtype: int
        """
        nums = [self.n_spans(id) for id in self.nodes()]
        if len(nums) > 0:
            return max(nums)
        else:
            return 1

    def n_gaps(self):
        """
        :return: Total number of gaps in any node.
        :rtype: int
        """
        return self.__n_gaps_below(self.virtual_root)

    def __n_gaps_below(self, id):
        """
        :param id: node id
        :type id: str
        :return: gaps below some node
        :rtype: int
        """
        n_gaps = self.n_spans(id) - 1
        for child in self.children(id):
            n_gaps += self.__n_gaps_below(child)
        return n_gaps

    def unlabelled_structure(self):
        """
        :return: pair consisting of (root and list of child nodes)
        :rtype: tuple[list[str], list]
        Create unlabelled structure, only in terms of breakup of yield
        """
        return self.unlabelled_structure_recur(self.virtual_root)

    def unlabelled_structure_recur(self, id):
        head = set(self.fringe(id))
        tail = [self.unlabelled_structure_recur(child) for child in self.children(id)]
        # remove useless step
        if len(tail) == 1 and head == tail[0][0]:
            return tail[0]
        else:
            return head, tail

    def recursive_partitioning(self):
        return self.recursive_partitioning_rec(self.virtual_root)
        # head = set(self.fringe(self.virtual_root))
        # tail = [self.recursive_partitioning_rec(root) for root in self.root]
        # if len(tail) == 1 and head == tail[0][0]:
        # return tail[0]
        # else:
        # return head, tail

    def recursive_partitioning_rec(self, id):
        head = set(self.fringe(id))
        if self.in_ordering(id):
            tail = [({self.node_index(id)}, [])]
        else:
            tail = []
        tail += map(self.recursive_partitioning_rec, self.children(id))
        if len(tail) == 1 and head == tail[0][0]:
            return tail[0]
        else:
            tail.sort(key=lambda elem: min(elem[0]))
            return head, tail

    def node_id_rec_par(self, rec_par):
        (head, tail) = rec_par
        head = list(map(lambda x: self.index_node(x + 1), head))
        tail = list(map(self.node_id_rec_par, tail))
        return head, tail

    def id_yield(self):
        """
        :return: list of node ids that are in the ordering and connected to root
        :rtype: list[str]
        """
        return self.__ordered_ids

    def full_yield(self):
        """
        :return: list of node ids that are in the ordering (including disconnected nodes)
        :rtype: list[str]
        """
        return self.__full_yield

    def token_yield(self):
        """
        :return: Get yield as list of all labels of nodes, that are in the ordering and connected to the root.
        :rtype: list[MonadicToken]
        """
        return [self.node_token(id) for id in self.__ordered_ids]

    def full_token_yield(self):
        """
        :return: Get yield as list of labels of nodes, that are in the ordering (including disconnected nodes).
        :rtype: list[MonadicToken]
        """
        return [self.node_token(id) for id in self.__full_yield]

    def nodes(self):
        """
        :return: ids of all nodes.
        :rtype: list[str]
        """
        return self._id_to_token.keys()

    def node_token(self, id):
        """
        :param id: node id
        :type id: str
        :return: token at node id
        :rtype: MonadicToken
        Query the token of node id.
        """
        return self._id_to_token[id]

    def complete(self):
        """
        :return: Does yield cover whole string?
        :rtype: bool
        """
        return len(self.fringe(self.virtual_root)) == len(self.__ordered_ids)

    def n_nodes(self):
        """
        :return: Number of nodes in tree that are connected to the root (or the root itself).
        :rtype: int
        """
        return self.__n_nodes_below(self.virtual_root)

    def __n_nodes_below(self, id):
        """
        :param id: node id
        :type id: str
        :return: Number of nodes below node.
        :rtype: int
        """
        n = len(self.children(id))
        for child in self.children(id):
            n += self.__n_nodes_below(child)
        return n

    def empty_fringe(self):
        """
        :rtype: bool
        Is there any non-ordered node without children?
        Includes the case the root has no children.
        """
        for id in self.nodes():
            if len(self.children(id)) == 0 and id not in self.full_yield():
                return True
        return len(self.fringe(self.virtual_root)) == 0

    def siblings(self, id):
        """
        :param id: node id
        :type id: str
        :return: list of node ids
        :rtype: list[str]
        The siblings of id, i.e. the children of id's parent (including id),
        ordered from left to right. If id is the root, then [root] is returned
        """
        if id in self.root:
            return self.root
        else:
            parent = self.parent(id)
            if not parent:
                raise Exception('non-root node has no parent!')
            return self.children(parent)

    def __hybrid_tree_str(self, root, level):
        my_string = self.node_token(root)
        my_string = str(my_string)
        s = level * ' ' + my_string + '\n'
        for child in self.children(root):
            s += self.__hybrid_tree_str(child, level + 1)
        return s

    def __str__(self):
        return ''.join([self.__hybrid_tree_str(id, 0) for id in self.root])

    def __eq__(self, other):
        if not isinstance(other, HybridTree):
            return False
        return all([self.compare_recursive(other, self_node, other_node) for self_node, other_node in
                    zip(self.root, other.root)])

    def __hash__(self):
        return hash((tuple(self.token_yield()), tuple([self.__hash_recursive(id) for id in self.root])))

    def __hash_recursive(self, id):
        return hash((self.node_token(id), tuple([self.__hash_recursive(child) for child in self.children(id)])))

    def compare_recursive(self, other, self_node, other_node):
        """
        Synchronously traverses two hybrid trees and compares the labels, pos-tags, deprels, position in ordering and
        number of children at each position.

        :param other:
        :type other: HybridTree
        :param self_node:
        :param other_node:
        :return:
        :rtype: bool
        """
        # Compare current nodes
        if not self.node_token(self_node).__eq__(other.node_token(other_node)):
            # print self.node_token(self_node), other.node_token(other_node)
            return False
        if self.in_ordering(self_node):
            if other.in_ordering(other_node):
                if self.node_index_full(self_node) != other.node_index(other_node):
                    return False
            else:
                return False
        elif other.in_ordering(other_node):
            return False

        # compare children of both nodes
        if not len(self.children(self_node)) == len(other.children(other_node)):
            return False

        children = [self.compare_recursive(other, self_child, other_child) for
                    self_child, other_child in zip(self.children(self_node), other.children(other_node))]

        return all(children)

    def reset_edge_labels(self, label):
        for node in self._id_to_token:
            self.node_token(node).set_edge_label(label)


class HybridDag(HybridTree):
    def __init__(self, sent_label):
        HybridTree.__init__(self, sent_label)
        self._id_to_sec_children = {}
        self._id_to_sec_child_labels = {}
        self._sec_parents = {}

    def add_sec_child(self, parent, child, edge_label):
        if parent in self._id_to_sec_children:
            self._id_to_sec_children[parent].append(child)
            self._id_to_sec_child_labels[parent].append(edge_label)
        else:
            self._id_to_sec_children[parent] = [child]
            self._id_to_sec_child_labels[parent] = [edge_label]

        if child in self._sec_parents and parent not in self._sec_parents[child]:
            self._sec_parents[child].append(parent)
        else:
            self._sec_parents[child] = [parent]

    def sec_children(self, node):
        return self._id_to_sec_children.get(node, [])

    def sec_child_edge_labels(self, node):
        return self._id_to_sec_child_labels.get(node, [])

    def sec_parents(self, node):
        return self._sec_parents.get(node, [])

    #TODO remove this code duplication, specialization concerning constituent case(code duplication ConstituentTree!)
    def labelled_spans(self):
        """
        :return: list of spans (each of which is string plus an even number of (integer) positions)
        Labelled spans.
        """
        spans = []
        for id in [n for n in self.nodes() if n not in self.full_yield()]:
            span = [self.node_token(id).category()]
            for (low, high) in join_spans(self.fringe(id)):
                span += [low, high]
            # TODO: this if-clause allows to handle trees, that have nodes with empty fringe
            if len(span) >= 3:
                spans += [span]
        return sorted(spans,
                      # cmp=lambda x, y: cmp([x[1]] + [-x[2]] + x[3:] + [x[0]], \
                      #                      [y[1]] + [-y[2]] + y[3:] + [y[0]])
                      key=lambda x: [tuple(x[1:]), x[0]])

    def topological_order(self, reverse=False):
        """
        :param reverse: reverse list
        :type reverse: bool
        :return: list of nodes of dag in topological order starting from leaves
        :rtype: list
        """
        order = []
        added = set()
        changed = True
        while changed:
            changed = False
            for node in self.nodes():
                if node in added:
                    continue
                if all([c in added for c in self.children(node) + self.sec_children(node)]):
                    added.add(node)
                    order.append(node)
                    changed = True
        if len(added) == len(self.nodes()):
            if reverse:
                return reversed(order)
            return order
        else:
            return None


__all__ = ["HybridTree"]