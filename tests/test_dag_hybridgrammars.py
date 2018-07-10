import unittest
import corpora.negra_parse as np
import corpora.tiger_parse as tp
from hybridtree.general_hybrid_tree import HybridDag
from hybridtree.monadic_tokens import construct_constituent_token
from constituent.dag_induction import direct_extract_lcfrs_from_prebinarized_corpus, top, bottom, \
    BasicNonterminalLabeling, PosTerminals
from grammar.induction.terminal_labeling import FormTerminals, CompositionalTerminalLabeling
from parser.naive.parsing import LCFRS_parser
from parser.discodop_parser.parser import DiscodopKbestParser
from parser.sDCPevaluation.evaluator import DCP_evaluator, dcp_to_hybriddag
import tempfile
import copy
import subprocess
import os
import json
import shutil
from grammar.lcfrs import LCFRS
from grammar.linearization import linearize
from graphs.schick_parser_rtg_import import read_rtg
from parser.supervised_trainer.trainer import PyDerivationManager
from grammar.lcfrs_derivation import LCFRSDerivationWrapper


SCHICK_PARSER_JAR = 'HypergraphReduct-1.0-SNAPSHOT.jar'


class DagHybridGrammarTest(unittest.TestCase):
    def test_negra_to_dag_parsing(self):
        names = list(map(str, [26954]))

        fd_, primary_file = tempfile.mkstemp(suffix='.export')
        with open(primary_file, mode='w') as pf:

            for s in names:
                dsg = tp.sentence_names_to_deep_syntax_graphs(["s" + s], "res/tiger/tiger_s%s.xml" % s, hold=False,
                                                              ignore_puntcuation=False)[0]
                dsg.set_label(dsg.label[1:])
                lines = np.serialize_hybrid_dag_to_negra([dsg], 0, 500, use_sentence_names=True)
                print(''.join(lines), file=pf)

        _, binarized_file = tempfile.mkstemp(suffix='.export')
        subprocess.call(["discodop", "treetransforms", "--binarize", "-v", "1", "-h", "1", primary_file, binarized_file])

        print(primary_file)
        print(binarized_file)

        corpus = np.sentence_names_to_hybridtrees(names, primary_file, secedge=True)
        corpus2 = np.sentence_names_to_hybridtrees(names, binarized_file, secedge=True)
        dag = corpus[0]
        print(dag)

        assert isinstance(dag, HybridDag)
        self.assertEqual(8, len(dag.token_yield()))
        for token in dag.token_yield():
            print(token.form() + '/' + token.pos(), end=' ')
        print()

        dag_bin = corpus2[0]
        print(dag_bin)

        for token in dag_bin.token_yield():
            print(token.form() + '/' + token.pos(), end=' ')
        print()
        self.assertEqual(8, len(dag_bin.token_yield()))

        for node, token in zip(dag_bin.nodes(), list(map(str, map(dag_bin.node_token, dag_bin.nodes())))):
            print(node, token)

        print()
        print(top(dag_bin, {'500', '101', '102'}))
        self.assertSetEqual({'101', '500'}, top(dag_bin, {'500', '101', '102'}))
        print(bottom(dag_bin, {'500', '101', '102'}))
        self.assertSetEqual({'502'}, bottom(dag_bin, {'500', '101', '102'}))

        nont_labeling = BasicNonterminalLabeling()
        term_labeling = FormTerminals()  # PosTerminals()

        grammar = direct_extract_lcfrs_from_prebinarized_corpus(dag_bin, term_labeling, nont_labeling)
        # print(grammar)

        for rule in grammar.rules():
            print(rule.get_idx(), rule)

        print("Testing LCFRS parsing and DCP evaluation".center(80, '='))

        parser = LCFRS_parser(grammar)

        parser_input = term_labeling.prepare_parser_input(dag_bin.token_yield())
        print(parser_input)
        parser.set_input(parser_input)

        parser.parse()

        self.assertTrue(parser.recognized())

        der = parser.best_derivation_tree()
        print(der)

        dcp_term = DCP_evaluator(der).getEvaluation()

        print(dcp_term[0])

        dag_eval = HybridDag(dag_bin.sent_label())
        dcp_to_hybriddag(dag_eval, dcp_term, copy.deepcopy(dag_bin.token_yield()), False, construct_token=construct_constituent_token)

        print(dag_eval)
        for node in dag_eval.nodes():
            token = dag_eval.node_token(node)
            if token.type() == "CONSTITUENT-CATEGORY":
                label = token.category()
            elif token.type() == "CONSTITUENT-TERMINAL":
                label = token.form(), token.pos()

            print(node, label, dag_eval.children(node), dag_eval.sec_children(node), dag_eval.sec_parents(node))

        lines = np.serialize_hybridtrees_to_negra([dag_eval], 1, 500, use_sentence_names=True)
        for line in lines:
            print(line, end='')

        print()

        with open(primary_file) as pcf:
            for line in pcf:
                print(line, end='')


        print('Testing reduct computation with Schick parser'.center(80, '='))

        grammar_path = '/tmp/lcfrs_dcp_grammar.gr'
        derivation_manager = PyDerivationManager(grammar)

        with open(grammar_path, 'w') as grammar_file:
            nonterminal_enc, terminal_enc = linearize(grammar, nont_labeling, term_labeling, grammar_file,
                                                      delimiter=' : ', nonterminal_encoder=derivation_manager.get_nonterminal_map())

        print(np.negra_to_json(dag, terminal_enc, term_labeling))
        json_data = np.export_corpus_to_json([dag], terminal_enc, term_labeling)

        corpus_path = '/tmp/json_dags.json'
        with open(corpus_path, 'w') as data_file:
            json.dump(json_data, data_file)

        reduct_dir = '/tmp/schick_parser_reducts'
        if os.path.isdir(reduct_dir):
            shutil.rmtree(reduct_dir)
        os.makedirs(reduct_dir)

        p = subprocess.Popen([' '.join(
            ["java", "-jar", os.path.join("util", SCHICK_PARSER_JAR), 'reduct', '-g', grammar_path, '-t',
             corpus_path, "--input-format", "json", "-o", reduct_dir])], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("stdout", p.stdout.name)

        while True:
            nextline = p.stdout.readline()
            if nextline == b'' and p.poll() is not None:
                break
            print(nextline.decode('unicode_escape'), end='')
            # sys.stdout.write(nextline)
            # sys.stdout.flush()

        p.wait()
        p.stdout.close()
        self.assertEqual(0, p.returncode)
        rtgs = []

        def decode_nonterminals(s):
            return derivation_manager.get_nonterminal_map().index_object(int(s))

        for i in range(1, len(corpus) + 1):
            rtgs.append(read_rtg(os.path.join(reduct_dir, str(i) + '.gra'), symbol_offset=-1, rule_prefix='r',
                                 process_nonterminal=decode_nonterminals))

        print("Reduct RTG")
        for rule in rtgs[0].rules:
            print(rule.lhs, "->", rule.symbol, rule.rhs)

        derivation_manager.get_nonterminal_map().print_index()
        derivation_manager.convert_rtgs_to_hypergraphs(rtgs)
        derivation_manager.serialize(bytes('/tmp/reduct_manager.trace', encoding='utf8'))
        derivations = [LCFRSDerivationWrapper(der) for der in derivation_manager.enumerate_derivations(0, grammar)]
        self.assertGreaterEqual(len(derivations), 1)

        if len(derivations) >= 1:
            print("Sentence", i)
            for der in derivations:
                print(der)
                self.assertTrue(der.check_integrity_recursive(der.root_id(), grammar.start()))

    def test_negra_dag_small_grammar(self):
        DAG_CORPUS = 'res/tiger/tiger_full_with_sec_edges.export'
        DAG_CORPUS_BIN = 'res/tiger/tiger_full_with_sec_edges_bin_h1_v1.export'
        names = list([str(i) for i in range(1, 101)])
        if not os.path.exists(DAG_CORPUS):
            print('run the following command to create an export corpus with dags:')
            print('\tPYTHONPATH=. util/tiger_dags_to_negra.py ' +
                  'res/tiger/tiger_release_aug07.corrected.16012013.xml '
                  + DAG_CORPUS + ' 1 50474')
        self.assertTrue(os.path.exists(DAG_CORPUS))

        if not os.path.exists(DAG_CORPUS_BIN):
            print('run the following command to binarize the export corpus with dags:')
            print("discodop treetransforms --binarize -v 1 -h 1 " + DAG_CORPUS + " " + DAG_CORPUS_BIN)
            # _, DAG_CORPUS_BIN = tempfile.mkstemp(prefix='corpus_bin_', suffix='.export')
            # subprocess.call(["discodop", "treetransforms", "--binarize", "-v", "1", "-h", "1", DAG_CORPUS, DAG_CORPUS_BIN])
        self.assertTrue(os.path.exists(DAG_CORPUS_BIN))
        corpus = np.sentence_names_to_hybridtrees(names, DAG_CORPUS, secedge=True)
        corpus_bin = np.sentence_names_to_hybridtrees(names, DAG_CORPUS_BIN, secedge=True)

        grammar = LCFRS(start="START")

        for hybrid_dag, hybrid_dag_bin in zip(corpus, corpus_bin):
            self.assertEqual(len(hybrid_dag.token_yield()), len(hybrid_dag_bin.token_yield()))

            dag_grammar = direct_extract_lcfrs_from_prebinarized_corpus(hybrid_dag_bin)
            grammar.add_gram(dag_grammar)

        grammar.make_proper()
        print("Extracted LCFRS/DCP-hybrid grammar with %i nonterminals and %i rules"
              % (len(grammar.nonts()), len(grammar.rules())))

        parser = DiscodopKbestParser(grammar, k=1)

        _, RESULT_FILE = tempfile.mkstemp(prefix='parser_results_', suffix='.export')

        with open(RESULT_FILE, 'w') as results:
            for hybrid_dag in corpus:

                poss = list(map(lambda x: x.pos(), hybrid_dag.token_yield()))
                parser.set_input(poss)
                parser.parse()
                self.assertTrue(parser.recognized())
                der = parser.best_derivation_tree()

                dcp_term = DCP_evaluator(der).getEvaluation()
                dag_eval = HybridDag(hybrid_dag.sent_label())
                dcp_to_hybriddag(dag_eval, dcp_term, copy.deepcopy(hybrid_dag.token_yield()), False,
                                 construct_token=construct_constituent_token)
                lines = np.serialize_hybridtrees_to_negra([dag_eval], 1, 500, use_sentence_names=True)
                for line in lines:
                    print(line, end='', file=results)
                parser.clear()

        print("Wrote results to %s" % RESULT_FILE)
        _, REFERENCE_FILE = tempfile.mkstemp(prefix='parser_reference_', suffix='.export')
        with open(REFERENCE_FILE, 'w') as ref:
            lines = np.serialize_hybridtrees_to_negra(corpus, 1, 500, use_sentence_names=True)
            for line in lines:
                print(line, end='', file=ref)
        print("Wrote reference corpus to %s" % REFERENCE_FILE)
        subprocess.call(["discodop", "eval", REFERENCE_FILE, RESULT_FILE, '--secedges=all'])


if __name__ == '__main__':
    unittest.main()
