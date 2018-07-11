from __future__ import print_function

from collections import defaultdict
from sys import stdout

from grammar.dcp import DCP_visitor
from grammar.lcfrs import LCFRS_var
from util.enumerator import Enumerator


def linearize(grammar, nonterminal_labeling, terminal_labeling, file, delimiter='::', nonterminal_encoder=None):
    """
    :type grammar: LCFRS
    :param nonterminal_labeling:
    :param terminal_labeling:
    :param file: file handle to write to
    :type delimiter: str
    :param delimiter: string used to join terminal symbol with edge label symbol
    :type nonterminal_encoder: Enumerator
    :param nonterminal_encoder: mapping that assigns unique non-negative integer to each nonterminal
    """
    print("Nonterminal Labeling: ", nonterminal_labeling, file=file)
    print("Terminal Labeling: ", terminal_labeling, file=file)
    print(file=file)

    terminals = Enumerator(first_index=1)
    if nonterminal_encoder is None:
        nonterminals = Enumerator()
    else:
        nonterminals = nonterminal_encoder
    num_inherited_args = {}
    num_synthesized_args = {}

    for rule in grammar.rules():
        rid = 'r%i' % (rule.get_idx() + 1)
        print(rid, 'RTG   ', nonterminals.object_index(rule.lhs().nont()), '->', file=file, end=" ")
        print(list(map(lambda nont: nonterminals.object_index(nont), rule.rhs())), ';', file=file)

        print(rid, 'WEIGHT', rule.weight(), ';', file=file)

        sync_index = {}
        inh_args = defaultdict(lambda: 0)
        lhs_var_counter = CountLHSVars()
        synthesized_attributes = 0

        dcp_ordered = sorted(rule.dcp(), key=lambda x: (x.lhs().mem(), x.lhs().arg()))

        for dcp in dcp_ordered:
            if dcp.lhs().mem() != -1:
                inh_args[dcp.lhs().mem()] += 1
            else:
                synthesized_attributes += 1
            lhs_var_counter.evaluate_list(dcp.rhs())
        num_inherited_args[nonterminals.object_index(rule.lhs().nont())] = inh_args[-1] = lhs_var_counter.get_number()
        num_synthesized_args[nonterminals.object_index(rule.lhs().nont())] = synthesized_attributes

        for dcp in dcp_ordered:
            printer = DcpPrinter(terminals.object_index, rule, sync_index, inh_args, delimiter=delimiter)
            printer.evaluate_list(dcp.rhs())
            var = dcp.lhs()
            if var.mem() == -1:
                var_string = 's<0,%i>' % (var.arg() + 1 - inh_args[-1])
            else:
                var_string = 's<%i,%i>' % (var.mem() + 1, var.arg() + 1)
            print('%s sDCP   %s == %s ;' % (rid, var_string, printer.string), file=file)

        s = 0
        for j, arg in enumerate(rule.lhs().args()):
            print(rid, 'LCFRS  s<0,%i> == [' % (j + 1), end=' ', file=file)
            first = True
            for a in arg:
                if not first:
                    print(",", end=' ', file=file)
                if isinstance(a, LCFRS_var):
                    print("x<%i,%i>" % (a.mem + 1, a.arg + 1), end=' ', file=file)
                    pass
                else:
                    if s in sync_index:
                        print(str(terminals.object_index(a)) + '^{%i}' % sync_index[s], end=' ', file=file)
                    else:
                        print(str(terminals.object_index(a)), end=' ', file=file)
                    s += 1
                first = False
            print('] ;', file=file)
        print(file=file)

    print("Terminals: ", file=file)
    terminals.print_index(to_file=file)
    print(file=file)

    print("Nonterminal ID, nonterminal name, fanout, #inh, #synth: ", file=file)
    max_fanout, max_inh, max_syn, max_args, fanouts, inherits, synths, args \
        = print_index_and_stats(nonterminals, grammar, num_inherited_args, num_synthesized_args, file=file)
    print(file=file)
    print("max fanout:", max_fanout, file=file)
    print("max inh:", max_inh, file=file)
    print("max synth:", max_syn, file=file)
    print("max args:", max_args, file=file)
    print(file=file)
    for s, d, m in [('fanout', fanouts, max_fanout), ('inh', inherits, max_inh),
                    ('syn', synths, max_syn), ('args', args, max_args)]:
        for i in range(m + 1):
            print('# the number of nonterminals with %s = %i is %i' % (s, i, d[i]), file=file)
        print(file=file)
    print(file=file)

    print("Initial nonterminal: ", nonterminals.object_index(grammar.start()), file=file)
    print(file=file)
    return nonterminals, terminals


def print_index_and_stats(nonterminal_encoder, grammar, inh, syn, file=stdout):
    fanouts = defaultdict(lambda: 0)
    inherited = defaultdict(lambda: 0)
    synthesized = defaultdict(lambda: 0)
    args = defaultdict(lambda: 0)
    max_fanout = 0
    max_inh = 0
    max_syn = 0
    max_args = 0
    for i in range(nonterminal_encoder.get_first_index(), nonterminal_encoder.get_counter()):
        fanout = grammar.fanout(nonterminal_encoder.index_object(i))
        fanouts[fanout] += 1
        max_fanout = max(max_fanout, fanout)
        inherited[inh[i]] += 1
        max_inh = max(max_inh, inh[i])
        synthesized[syn[i]] += 1
        max_syn = max(max_syn, syn[i])
        args[inh[i] + syn[i]] += 1
        max_args = max(max_args, inh[i] + syn[i])
        print(i, nonterminal_encoder.index_object(i), fanout, inh[i], syn[i], file=file)
    return max_fanout, max_inh, max_syn, max_args, fanouts, inherited, synthesized, args


class DcpLabelCollector(DCP_visitor):
    def visit_string(self, s, idx):
        self.labels.add(s)

    def visit_term(self, term, idx):
        """
        :type term: DCP_term
        :param idx:
        :return:
        """
        term.head().visitMe(self)
        for child in term.arg():
            child.visitMe(self)

    def visit_index(self, index, idx):
        """
        :type index: DCP_index
        :param idx:
        :return:
        """
        self.labels.add(index.edge_label())

    def visit_variable(self, var, idx):
        pass

    def __init__(self):
        self.labels = set()


class CountLHSVars(DCP_visitor):
    def __init__(self):
        self.__vars = set()

    def visit_variable(self, var, idx):
        if var.mem() == -1:
            self.__vars.add(var)

    def visit_string(self, s, idx):
        pass

    def visit_term(self, term, idx):
        term.head().visitMe(self)
        self.evaluate_list(term.arg())

    def evaluate_list(self, xs):
        for x in xs:
            x.visitMe(self)

    def visit_index(self, index, idx):
        pass

    def get_number(self):
        return len(self.__vars)


class DcpPrinter(DCP_visitor):
    def visit_variable(self, var, idx):
        if var.mem() != -1:
            self.string += 'x<%i,%i> ' % (var.mem() + 1, var.arg() + 1 - self.inh_args[var.mem()])
        else:
            self.string += 'x<%i,%i> ' % (var.mem() + 1, var.arg() + 1)

    def __init__(self, terminal_to_index, rule, sync_index, inh_args, delimiter='::'):
        self.terminal_to_index = terminal_to_index
        self.rule = rule
        self.string = ''
        self.sync_index = sync_index
        self.inh_args = inh_args
        self.delimiter = delimiter

    def visit_string(self, s, idx):
        if s.edge_label():
            self.string += str(self.terminal_to_index(s.get_string() + self.delimiter + s.edge_label())) + ' '
        else:
            self.string += str(self.terminal_to_index(s.get_string()))

    def visit_term(self, term, idx):
        term.head().visitMe(self)
        self.string += '('
        self.evaluate_list(term.arg())
        self.string += ') '

    def evaluate_list(self, the_list):
        self.string += "[ "
        first = True
        for arg in the_list:
            if not first:
                self.string += ', '
            arg.visitMe(self)
            first = False
        self.string += "]"

    def visit_index(self, index, idx):
        if not index.index() in self.sync_index:
            self.sync_index[index.index()] = len(self.sync_index) + 1
        i = 0
        for arg in self.rule.lhs().args():
            for obj in arg:
                if not isinstance(obj, LCFRS_var):
                    if i == index.index():
                        self.string += "%i^{%i}" % (self.terminal_to_index(obj + self.delimiter + index.edge_label()),
                                                    self.sync_index[index.index()])
                        return
                    else:
                        i += 1
