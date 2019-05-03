from __future__ import print_function, unicode_literals
from grammar.lcfrs import LCFRS, LCFRS_lhs, LCFRS_var
from discodop.tree import escape
import re


def escape_brackets(nont):
    return nont.replace("(", "__OB__").replace(")", "__CB__")


def unescape_brackets(nont):
    return nont.replace("__OB__", "(").replace("__CB__", ")")


def transform_grammar(grammar):
    """
    :type grammar: LCFRS
    """
    # TODO assert terminals only in rules with len(rhs) = 0
    for rule in grammar.rules():
        assert rule.ordered()
        if rule.weight() == 0.0:
            continue
        fake_nont = escape_brackets(rule.lhs().nont()) + "-" + str(rule.get_idx())
        trans_rule_fake = (escape_brackets(rule.lhs().nont()), fake_nont), tuple([(0,) for _ in rule.lhs().args()])
        yield trans_rule_fake, rule.weight()
        rhs = list(map(escape_brackets, rule.rhs())) if rule.rhs() else ['Epsilon']
        trans_rule = tuple([fake_nont] + rhs), transform_args(rule.lhs().args())
        yield trans_rule, 1.0


def transform_args(args):
    def transform_arg(arg):
        arg_new = []
        for elem in arg:
            if isinstance(elem, LCFRS_var):
                arg_new.append(elem.mem)
            else:
                assert len(arg) == 1
                return escape(elem)
        return tuple(arg_new)
    return tuple([transform_arg(arg) for arg in args])


def transform_grammar_cfg_approx(grammar):
    """
    :type grammar: LCFRS
    """
    # TODO assert terminals only in rules with len(rhs) = 0
    for rule in grammar.rules():
        assert rule.ordered()
        if rule.weight() == 0.0:
            continue
        for n, arg in enumerate(rule.lhs().args()):
            appendix = "*" + str(n) if rule.lhs().fanout() > 1 else ""
            fake_nont = escape_brackets(rule.lhs().nont()) + "-" + str(rule.get_idx()) + appendix
            trans_rule_fake = (escape_brackets(rule.lhs().nont()) + appendix, fake_nont), ((0,),)
            yield trans_rule_fake, rule.weight()
            # todo: rules fanout > 1 get suppressed in CFG chart because they are accounted for
            # todo: with rule.weight()^fanout. This alternative (is inproper) but doesn't work well.
            # yield trans_rule_fake, pow(rule.weight(), 1.0 / rule.lhs().fanout())

            for lhs, transformed_arg, rhs in \
                    transform_args_to_bin_cfg(fake_nont, arg, rule.rhs(), grammar):
                trans_rule = tuple([lhs] + rhs), (transformed_arg,)
                yield trans_rule, 1.0


def transform_args_to_bin_cfg(lhs, arg, rhs, grammar):
    arg_new = []
    rhs_new = []
    for elem in arg:
        if isinstance(elem, LCFRS_var):
            arg_new.append(len(rhs_new))
            appendix = "*" + str(elem.arg) if grammar.fanout(rhs[elem.mem]) > 1 else ""
            rhs_new.append(escape_brackets(rhs[elem.mem]) + appendix)
        else:
            assert len(arg) == 1
            yield lhs, escape(elem), ['Epsilon']
            return

    assert rhs_new != []
    if len(rhs_new) <= 2:
        yield lhs, tuple(arg_new), rhs_new
    else:
        lhs_bar = lhs + "<>BAR"
        generic_yf = (0, 1)
        for i, rhs_nont in enumerate(rhs_new[:-2]):
            if i == 0:
                yield lhs, generic_yf, [rhs_nont, lhs_bar + "<>" + rhs_nont]
            else:
                yield lhs_bar + "<>" + rhs_new[i - 1], generic_yf, [rhs_nont, lhs_bar + "<>" + rhs_nont]

        yield lhs_bar + "<>" + rhs_new[-3], generic_yf, [rhs_new[-2], rhs_new[-1]]


striplabelre = re.compile(r'^(.*)-(\d+)$')


def rule_idx_from_label(label):
    split = striplabelre.split(label)
    assert len(split) == 4
    rule_idx = int(split[-2])
    return rule_idx


__all__ = ["rule_idx_from_label", "transform_grammar", "transform_grammar_cfg_approx"]
