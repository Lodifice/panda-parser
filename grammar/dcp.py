# Definite clause program rules. A list of such rules is
# part of a LCFRS/DCP hybrid grammar rule.

from __future__ import print_function
from abc import ABCMeta, abstractmethod

###########################################################################
# Parts of the rules.


# Common interface for all objects that occur on rhs of DCP_rules
class DCP_rhs_object:
    __metaclass__ = ABCMeta

    # evaluator: DCP_visitor
    # id: string (gorn term of LCFRS-Derivation tree)
    @abstractmethod
    def visitMe(self, visitor, id=None):
        """
        :type visitor: DCP_visitor
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: evaluated DCP_rhs object
        """
        pass


# Interface for DCP_evaluation
class DCP_visitor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_string(self, s, id):
        """
        :type s: DCP_string
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: evaluated DCP_string
        :rtype: DCP_string
        """
        pass

    @abstractmethod
    def visit_index(self, index, id):
        """
        :type index: DCP_index
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: the input position to which the index points
        :rtype: DCP_position
        """
        pass

    @abstractmethod
    def visit_term(self, term, id):
        """
        :type term: DCP_term
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: evaluated DCP_term
        :rtype: DCP_term
        """
        pass

    @abstractmethod
    def visit_variable(self, var, id):
        pass


# Variable identifying argument (synthesized or inherited).
# In LHS this is (-1,j) and in RHS this is (i,j),
# for i-th member in RHS, and j-th argument.
class DCP_var(DCP_rhs_object):
    # Constructor.
    # i: int
    # j: int
    def __init__(self, i, j):
        self.__i = i
        self.__j = j

    # Member number part of variable, or -1 for LHS.
    # return: int
    def mem(self):
        return self.__i

    # Argument number part of variable.
    # return: int
    def arg(self):
        return self.__j

    # String representation.
    # return: string
    def __str__(self):
        if self.mem() < 0:
            return '<' + str(self.arg()) + '>'
        else:
            return '<' + str(self.mem()) + ',' + str(self.arg()) + '>'

    def visitMe(self, visitor, id=None):
        return visitor.visit_variable(self, id)

    def __eq__(self, other):
        if not isinstance(other, DCP_var):
            return False
        return self.mem() == other.mem() and self.arg() == other.arg()

    def __hash__(self):
        return hash((self.__i, self.__j))


# Index, pointing to terminal in left (LCFRS) component of hybrid grammar.
# Terminals are indexed in left-to-right order.
class DCP_index(DCP_rhs_object):
    # Constructor.
    # i: int
    # edge_label: string
    def __init__(self, i, edge_label=None):
        self.__i = i
        self.__edge_label = edge_label

    # The index.
    # return: int
    def index(self):
        return self.__i

    def edge_label(self):
        return self.__edge_label

    # String representation.
    # return: string
    def __str__(self):
        if self.__edge_label:
            s = ':{' + self.__edge_label + '}'
        else:
            s = ''
        return '[' + str(self.index()) + s + ']'

    # Evaluator Invocation
    def visitMe(self, visitor, id=None):
        return visitor.visit_index(self, id)


# A terminal of DCP_rule that is not linked to some terminal
# in the LCFRS component of the hybrid grammar
class DCP_string(DCP_rhs_object):
    def __init__(self, string, edge_label=None):
        self.__string = string
        self.__edge_label = edge_label

    def set_edge_label(self, edge_label):
        self.__edge_label = edge_label

    def edge_label(self):
        return self.__edge_label

    # Evaluator invocation
    def visitMe(self, visitor, id=None):
        return visitor.visit_string(self, id)

        # String representation.
        # return: string

    def get_string(self):
        return self.__string

    def __str__(self):
        if self.__edge_label:
            s = ':{' + self.__edge_label + '}'
        else:
            s = ''
        return self.__string + s


# An index replaced by an input position, according to parsing of a string with
# the left (LCFRS) component of hybrid grammar.
class DCP_position:
    # Constructor.
    # pos: int
    # edge_label: string
    def __init__(self, position, edge_label=None):
        self.__position = position
        self.__edge_label = edge_label

    # The position.
    # return: int
    def position(self):
        return self.__position

    def edge_label(self):
        return self.__edge_label

    # String representation.
    # return: string
    def __str__(self):
        if self.__edge_label:
            s = ':{' + self.__edge_label + '}'
        else:
            s = ''
        return '[' + str(self.position()) + s + ']'


# A terminal occurrence (may linked to LCFRS terminal),
# consisting of a DCP_string or DCP_index and a list of child terminal
# occurrences.
class DCP_term(DCP_rhs_object):
    # Constructor.
    # head: DCP_rhs_object (DCP_string / DCP_index)
    # arg: list of DCP_term/DCP_index TODO: outdated
    # arg: list of DCP_rhs_object (DCP_term + DCP_var)
    def __init__(self, head, arg):
        self.__head = head
        self.__arg = arg

    # The label.
    # return: string
    def head(self):
        return self.__head

    # The children.
    # return: list of DCP_term/DCP_index TODO: outdated
    # return: list of DCP_rhs_object (DCP_term / DCP_var)
    def arg(self):
        return self.__arg

    # String representation.
    # return: string
    def __str__(self):
        return str(self.head()) + '(' + dcp_terms_to_str(self.arg()) + ')'

    # Evaluator invocation
    def visitMe(self, visitor, id=None):
        return visitor.visit_term(self, id)


# Rule defining argument value by term.
class DCP_rule:
    def __init__(self, lhs, rhs):
        """
        :type lhs: DCP_var
        :type rhs: list(DCP_rhs_object)
        """
        self.__lhs = lhs
        self.__rhs = rhs

    # The LHS.
    # return: DCP_var
    def lhs(self):
        return self.__lhs

    # The RHS.
    # return: list of DCP_term/DCP_index TODO: outdated
    # return: list of DCP_rhs_object
    def rhs(self):
        return self.__rhs

    # String representation.
    # return: string
    def __str__(self):
        return str(self.lhs()) + '=' + dcp_terms_to_str(self.rhs())


################################################################
# Auxiliary.

# Turn list of terms into string. The terms are separated by whitespace.
# l: list of DCP_term/DCP_index TODO: outdated
# l: list of DCP_rhs_object
#
# return: string
def dcp_terms_to_str(l):
    return ' '.join([str(o) for o in l])


# Turn list of DCP_rules into string. The rules are separated by semicolons.
# l: list of DCP_rule
# return: string
def dcp_rules_to_str(l):
    return '; '.join([str(r) for r in l])


# As above, but more compact, omitting whitespace.
# l: list of DCP_rule
# return: string
def dcp_rules_to_key(l):
    return ';'.join([str(r) for r in l])
