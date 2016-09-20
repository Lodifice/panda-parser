from pynini import *
from grammar.LCFRS.lcfrs import LCFRS
from grammar.linearization import Enumerator
from math import log, e
from parser.derivation_interface import AbstractDerivation
from parser.parser_interface import AbstractParser

FINAL = 'THE-FINAL-STATE'

def compile_wfst_from_right_branching_grammar(grammar):
    """
    :type grammar: LCFRS
    :rtype: Fst
    Create a FST from a right-branching hybrid grammar.
    The Output of the is a rule tree in `polish notation <https://en.wikipedia.org/wiki/Polish_notation>`_
    """
    myfst = Fst()

    nonterminals = SymbolTable()
    for nont in grammar.nonts():
        sid = myfst.add_state()
        nonterminals.add_symbol(nont, sid)
        if nont == grammar.start():
            myfst.set_start(sid)
    sid = myfst.add_state()
    nonterminals.add_symbol(FINAL, sid)

    myfst.set_final(nonterminals.add_symbol(FINAL))

    rules = Enumerator(first_index=1)
    for rule in grammar.rules():
        rules.object_index(rule)

    terminals = SymbolTable()
    terminals.add_symbol('<epsilon>', 0)

    for rule in grammar.rules():
        if len(rule.rhs()) == 2:
            for rule2 in grammar.lhs_nont_to_rules(rule.rhs_nont(0)):
                if len(rule2.rhs()) == 0:
                    arc = Arc(terminals.add_symbol(rule2.lhs().args()[0][0]),
                              terminals.add_symbol(str(rules.object_index(rule))
                                                     + '-' + str(rules.object_index(rule2))),
                              make_weight(rule.weight() * rule2.weight()),
                              nonterminals.find(rule.rhs_nont(1)))
                    myfst.add_arc(nonterminals.find(rule.lhs().nont()), arc)
        elif len(rule.rhs()) == 0:
            arc = Arc(terminals.add_symbol(rule.lhs().args()[0][0]),
                      terminals.add_symbol(str(rules.object_index(rule))), make_weight(rule.weight()), nonterminals.find(FINAL))
            myfst.add_arc(nonterminals.find(rule.lhs().nont()), arc)
        else:
            assert rule.lhs().nont() == grammar.start()
            arc = Arc(0, terminals.add_symbol(str(rules.object_index(rule))), make_weight(rule.weight()), nonterminals.find(rule.rhs_nont(0)))
            myfst.add_arc(myfst.start(), arc)

    myfst.set_input_symbols(terminals)
    myfst.set_output_symbols(terminals)

    myfst.optimize(True)

    return myfst, rules


def fsa_from_list_of_symbols2(input):
    return acceptor(''.join(['[' + s + ']' for s in input]))


def fsa_from_list_of_symbols(input, symbol_table):
    """
    :param input:
    :type input:
    :param symbol_table:
    :type symbol_table: SymbolTable
    :return: An acceptor for the given list of tokens.
    :rtype: Fst
    The symbol table gets extended, if new tokens occur in the input.
    """
    fsa = Fst()
    fsa.set_input_symbols(symbol_table)
    fsa.set_output_symbols(symbol_table)
    state = fsa.add_state()
    fsa.set_start(state)
    for x in input:
        next_state = fsa.add_state()
        try:
            arc = Arc(symbol_table.find(x), symbol_table.find(x), 0, next_state)
        except KeyError:
            arc = Arc(symbol_table.add_symbol(x), symbol_table.add_symbol(x), 0, next_state)
        fsa.add_arc(state, arc)
        state = next_state
    fsa.set_final(state)
    return fsa


def make_weight(weight):
    return -log(weight)


def retrieve_rules(linear_fst, rpn=False):
    if rpn:
        linear_fst = reverse(rpn)
    linear_rules = []
    terminals = linear_fst.output_symbols()
    for s in range(linear_fst.num_states()):
        for arc in linear_fst.arcs(s):
            lab = terminals.find(arc.olabel)
            if isinstance(lab, str):
                linear_rules += [int(rule_string) for rule_string in lab.split("-")]
            else:
                linear_rules += [lab]
    return linear_rules


class PolishDerivation(AbstractDerivation):
    def child_ids(self, id):
        if id % 2 == 1 or id == self._len-1:
            return []
        else:
            return [id + 1, id + 2]


    def getRule(self, id):
        if id >= self._len:
            print
            print id
            print self._len
        return self._rule_list[id]

    def __str__(self):
        return self.der_to_str_rec(self.root_id(), 0)

    def der_to_str_rec(self, item, indentation):
        s = ' ' * indentation * 2 + str(self.getRule(item)) + '\t(' + str(item) + ')\n'
        for child in self.child_ids(item):
            s += self.der_to_str_rec(child, indentation + 1)
        return s

    def child_id(self, id, i):
        return id + i + 1

    def position_relative_to_parent(self, id):
        return id - 2 + (id % 2), (id + 1) % 2

    def root_id(self):
        return 0

    def terminal_positions(self, id):
        if id % 2 == 1 or id == self._len - 1:
            return [id / 2 + 1]
        else:
            return []

    def ids(self):
        return self._ids

    def __init__(self, rule_list):
        self._rule_list = rule_list
        self._len = len(rule_list)
        self._ids = range(self._len)


class RightBranchingFSTParser(AbstractParser):
    def recognized(self):
        pass

    def best_derivation_tree(self):
        polish_rules = retrieve_rules(self._best, None)
        if polish_rules:
            polish_rules = map(self._rules.index_object, polish_rules)
            der = PolishDerivation(polish_rules[1::])
            return der
        else:
            return None

    def __init__(self, grammar, input):
        fst, self._rules = grammar.tmp
        fsa = fsa_from_list_of_symbols(input, fst.mutable_input_symbols())

        intersection = fsa * fst

        self._best = best = shortestpath(intersection)
        best.topsort()

    def best(self):
        return pow(e, -float(shortestdistance(self._best)[-1]))

    def all_derivation_trees(self):
        pass

    @staticmethod
    def preprocess_grammar(grammar):
        grammar.tmp = compile_wfst_from_right_branching_grammar(grammar)