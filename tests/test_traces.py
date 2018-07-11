import unittest
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.trace_manager.sm_trainer_util import PyGrammarInfo
from util.enumerator import Enumerator
from grammar.lcfrs import LCFRS, LCFRS_lhs, LCFRS_var
from grammar.rtg import RTG


class TraceManagerTest(unittest.TestCase):
    def build_grammar(self):
        grammar = LCFRS("S")

        lhs1 = LCFRS_lhs("S")
        lhs1.add_arg([LCFRS_var(0, 0), LCFRS_var(1, 0)])
        rule_1 = grammar.add_rule(lhs1, ["S", "S"])

        lhs2 = LCFRS_lhs("S")
        lhs2.add_arg(["a"])
        rule_2 = grammar.add_rule(lhs2, [])

        lhs3 = LCFRS_lhs("A")
        lhs3.add_arg(["a"])
        rule_3 = grammar.add_rule(lhs3, [])

        return grammar, rule_1.get_idx(), rule_2.get_idx()

    def test_something(self):
        grammar, r1, r2 = self.build_grammar()
        nont_map = Enumerator()
        grammar_info = PyGrammarInfo(grammar, nont_map)

        def w(x):
            return "S", x

        rtg = RTG(w(3))
        rtg.construct_and_add_rule(w(3), r1, [w(1), w(2)])
        rtg.construct_and_add_rule(w(3), r1, [w(2), w(1)])
        rtg.construct_and_add_rule(w(2), r1, [w(1), w(1)])
        rtg.construct_and_add_rule(w(1), r2, [])

        rtg2 = RTG(("A", 3))

        rtg3 = RTG(w(3))
        rtg3.construct_and_add_rule(w(3), r1, [w(1), w(2)])
        rtg3.construct_and_add_rule(w(3), r1, [w(2), w(1)])
        rtg3.construct_and_add_rule(w(2), r2, [w(1), w(1)])
        rtg3.construct_and_add_rule(w(1), r2, [])

        traces = PyDerivationManager(grammar, nont_map)
        traces.convert_rtgs_to_hypergraphs([rtg, rtg2, rtg3])

        self.assertTrue(traces.is_consistent_with_grammar(grammar_info, traceId=0))
        self.assertFalse(traces.is_consistent_with_grammar(grammar_info, traceId=1))
        self.assertFalse(traces.is_consistent_with_grammar(grammar_info, traceId=2))


if __name__ == '__main__':
    unittest.main()
