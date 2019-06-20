from __future__ import print_function

import copy
import os
import pickle
import re
import subprocess
import time
try:
    from string import maketrans
except ModuleNotFoundError:
    maketrans = str.maketrans

import plac

import dependency.induction as d_i
import dependency.labeling as d_l
import grammar.induction.recursive_partitioning
import grammar.induction.terminal_labeling
from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from dependency.minimum_risk import compute_minimum_risk_tree
from dependency.oracle import compute_oracle_tree
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
from parser.LCFRS.LCFRS_trace_manager import compute_LCFRS_reducts, PyLCFRSTraceManager
from parser.coarse_to_fine_parser.coarse_to_fine import Coarse_to_fine_parser
from parser.parser_factory import GFParser, GFParser_k_best, CFGParser, LeftBranchingFSTParser, RightBranchingFSTParser
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
from parser.sDCPevaluation.evaluator import dcp_to_hybridtree, DCP_evaluator
from parser.trace_manager.score_validator import PyCandidateScoreValidator
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager
from parser.trace_manager.sm_trainer import PySplitMergeTrainerBuilder, \
    build_PyLatentAnnotation_initial, build_PyLatentAnnotation
from parser.supervised_trainer.trainer import PyDerivationManager
from playground_rparse.process_rparse_grammar import fall_back_left_branching

start = 'START'
dir = 'exp12/'

import hashlib
import sys


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


def compute_grammar_name(prefix, grammar_id, suffix):
    return os.path.join(prefix, grammar_id + '-' + suffix + '.pkl')


def grammar_id(corpus, nonterminal_labeling, terminal_labeling, recursive_partitioning):
    name = ''
    train_checksum = sha256_checksum(corpus._path)
    train_suffix = os.path.split(corpus._path)[1].split('.conll')[0]
    name += train_suffix
    if (corpus._start > 0):
        name += '-' + str(corpus._start)
    name += '-' + str(corpus._end)
    name += '-' + str(nonterminal_labeling)
    name += '-' + str(terminal_labeling)
    name += '-' + str('-'.join([rp.__name__ for rp in recursive_partitioning]))
    name += '-' + str(train_checksum)
    name = name.translate(maketrans('/', '-'))
    return name


def compute_reduct_name(prefix, grammar_id, corpus, suffix=''):
    reduct = grammar_id
    reduct += '-' + str(os.path.split(corpus._path)[1].split('.conll')[0])
    reduct += '-' + str(corpus._start)
    reduct += '-' + str(corpus._end)
    reduct += '-reducts' + suffix + '.hygra'
    return os.path.join(prefix, reduct)


def compute_sm_info_path(dir, baseline_id, emEpochs, rule_smoothing, splitRandomization, seed, discr, validation, corpus_validation, init):
    name = ['sm_info', str(emEpochs), str(rule_smoothing), str(splitRandomization), str(seed),
            'discr' if discr else '', str(init)]
    if validation:
        validation_checksum = sha256_checksum(corpus_validation._path)
        name += [validation_checksum, corpus_validation._start, corpus_validation._end]

    return os.path.join(dir, baseline_id, '-'.join(map(str, name)) + '.pkl')


def compute_sm_grammar_id(baseline_id, emEpochs, rule_smoothing, splitRandomization, seed, discr, validation, corpus_validation, init, cycle):
    name = ['sm_grammar', str(emEpochs), str(rule_smoothing), str(splitRandomization), str(seed),
            'discr' if discr else '', str(init), 'cycle_' + str(cycle)]
    if validation:
        validation_checksum = sha256_checksum(corpus_validation._path)
        name += [validation_checksum, corpus_validation._start, corpus_validation._end]

    return os.path.join(baseline_id, '-'.join(map(str, name)))


# baseline_path = dir + 'baseline_grammar.pkl'
# reduct_path = dir + 'reduct.pkl'
# reduct_path_discr = dir + 'reduct_discr.pkl'
# sm_info_path = dir + 'sm_info.pkl'
def em_trained_path(prefix, grammar_id, n_epochs, init, tie_breaking, seed):
    return os.path.join(prefix, grammar_id + '-' + str(n_epochs) + '_' + init + ('_tie_breaking_' if tie_breaking else '') + '-seed_' + str(seed) + '.pkl')


def sm_path(prefix, grammar_id, cycles):
    return os.path.join(prefix, grammar_id, 'sm_' + str(cycles) + '_grammar.pkl')


class Corpus:
    def __init__(self, path, type="CONLL", start=0, end=None, max_length=None, caching=True):
        self._path = path
        self._type = type
        self._start = start
        self._end = sys.maxint if end is None else end
        self._max_length = sys.maxint if max_length is None else max_length
        self._caching = caching
        self._trees = None

    def get_trees(self):
        if self._trees is not None:
            for tree in self._trees:
                yield tree
        else:
            self._trees = []
            for tree in length_limit(parse_conll_corpus(self._path, False, limit=self._end, start=self._start), self._max_length):
                self._trees.append(tree)
                yield tree


# term_labelling =  #d_i.the_terminal_labeling_factory().get_strategy('pos')
# recursive_partitioning = d_i.the_recursive_partitioning_factory().getPartitioning('fanout-1')
# recursive_partitioning = d_i.the_recursive_partitioning_factory().getPartitioning('left-branching')
primary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')
secondary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('strict', 'deprel')
ternary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'deprel')
child_top_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('childtop', 'deprel')
empty_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('empty', 'pos')

ignore_punctuation=False


@plac.annotations(
      recompileGrammar=('force (repeated) grammar induction and compilation', 'option', None, str)
    , retrain=('repeat weight training', 'flag')
    , limit=('training sentence limit', 'option', None, int)
    , max_length=('training sentence length limit', 'option', None, int)
    , dir=('directory to run the experiment in', 'option')
    , train=('path/to/training/corpus (used for training and validation)', 'option')
    , test=('path/to/test/corpus (used for testing)', 'option')
    , seed=('initial random seed', 'option', None, int)
    , emEpochs=('maximum number of epochs in EM training', 'option', None, int)
    , emInit=('initial weights for EM training (rfe ~ times induced)', 'option', None, str, ['rfe', 'equal'])
    , emTieBreaking=('perturbate initial weights before EM training', 'option')
    , splitRandomization=('percentage by which weights are pertubated after splitting', 'option', None, float)
    , mergePercentage=('merge percentage of splits', 'option', None, float)
    , smCycles=('total number of split/merge cycles', 'option', None, int)
    , validation=('use validation corpus', 'option', None, bool)
    , validationMethod=('score used for validation', 'option', None, str, ['likelihood', 'LAS', 'UAS'])
    , validationCorpus=('path/to/validation/corpus', 'option')
    , validationSplit=('use last percentage of sentences of training corpus for validation, if no validation corpus is specified', 'option', None, float)
    , validationDropIterations=("number of successive epochs of EM training, in which validation likelihood may drop", 'option', None, int)
    , parsing=('run parser on test corpus', 'option', None, bool, [True, False])
    , reparse=('rerun parser if result file already exists', 'flag')
    , parser=('parser engine', 'option', None, str, ["CFG", "GF", "GF-k-best", "FST", "CoarseToFine"])
    , test_limit=('test sentence limit', 'option', None, int)
    , k_best=('set length of k best length (GF-k-best parser only)', 'option', None, int)
    , minimum_risk=('compute a minimum risk tree from k-best list (GF-k-best parser only)', 'flag', "mr")
    , oracle_parse=('select the oracle tree from k-best list (GF-k-best/CoarseToFine parser only)', 'flag', "op")
    , recursive_partitioning=('rec. part. strategy', 'option', 'rec_par', str, ["cfg", "left-branching", "right-branching"] + ["fanout-" + str(i) for i in range(1, 9)])
    , nonterminal_labeling=('nonterminal labeling strategy', 'option', None, str)
    , terminal_labeling=('terminal labeling strategy', 'option', None, str)
    , rule_pruning=('prune rules with smaller prob., when compiling grammar for parsing', 'option')
    , rule_smoothing=('smooth rules with factor, when compiling grammar for parsing', 'option')
    , discr=('use conditional likelihood as EM optimization criterion (often infeasible), implemented by scaling rule counts', 'option')
    , maxScaleDiscr=('truncate scaling factor for rule counts', 'option')
)
def main(limit=3000
         , test_limit=sys.maxint
         , max_length=sys.maxint
         , dir=dir
         , train ='../res/negra-dep/negra-lower-punct-train.conll'
         , test = '../res/negra-dep/negra-lower-punct-test.conll'
         , recursive_partitioning='cfg'
         , nonterminal_labeling='childtop-deprel'
         , terminal_labeling='form-unk-30/pos'
         , emEpochs=20
         , emTieBreaking=True
         , emInit="rfe"
         , splitRandomization=1.0
         , mergePercentage=85.0
         , smCycles=6
         , rule_pruning=0.0001
         , rule_smoothing=0.01
         , validation=True
         , validationMethod = 'likelihood'
         , validationCorpus=None
         , validationSplit=20
         , validationDropIterations=6
         , seed=1337
         , discr=False
         , maxScaleDiscr=10
         , recompileGrammar="True"
         , retrain=False
         , parsing=True
         , reparse=False
         , parser="CFG"
         , k_best=50
         , minimum_risk=False
         , oracle_parse=False
         ):

    # set various parameters
    recompileGrammar = True if recompileGrammar == "True" else False
    # print(recompileGrammar)

    def result(gram, add=None):
        if add is not None:
            return os.path.join(dir, gram + '_experiment_parse_results_' + add + '.conll')
        else:
            return os.path.join(dir, gram + '_experiment_parse_results.conll')

    recursive_partitioning = grammar.induction.recursive_partitioning.the_recursive_partitioning_factory().get_partitioning(recursive_partitioning)
    top_level, low_level = tuple(nonterminal_labeling.split('-'))
    nonterminal_labeling = d_l.the_labeling_factory().create_simple_labeling_strategy(top_level, low_level)

    if parser == "CFG":
        assert all([rp.__name__  in ["left_branching", "right_branching", "cfg", "fanout_1"] for rp in recursive_partitioning])
        parser = CFGParser
    elif parser == "GF":
        parser = GFParser
    elif parser == "GF-k-best":
        parser = GFParser_k_best
    elif parser == "CoarseToFine":
        parser = Coarse_to_fine_parser
    elif parser == "FST":
        if recursive_partitioning == "left_branching":
            parser = LeftBranchingFSTParser
        elif recursive_partitioning == "right_branching":
            parser = RightBranchingFSTParser
        else:
            assert False and "expect left/right branching recursive partitioning for FST parsing"

    if validation:
        if validationCorpus is not None:
            corpus_validation = Corpus(validationCorpus)
            train_limit = limit
        else:
            train_limit = int(limit * (100.0 - validationSplit) / 100.0)
            corpus_validation = Corpus(train, start=train_limit, end=limit)
    else:
        train_limit = limit

    corpus_induce = Corpus(train, end=limit)
    corpus_train = Corpus(train, end=train_limit)
    corpus_test = Corpus(test, end=test_limit)

    match = re.match(r'^form-unk-(\d+)-morph.*$', terminal_labeling)
    if match:
        unk_threshold = int(match.group(1))
        term_labelling = grammar.induction.terminal_labeling.FormPosTerminalsUnkMorph(corpus_induce.get_trees(), unk_threshold, pos_filter=["NE", "CARD"],
                                                                                      add_morph={'NN': ['case', 'number', 'gender']
                        # , 'NE': ['case', 'number', 'gender']
                        # , 'VMFIN': ['number', 'person']
                        # , 'VVFIN': ['number', 'person']
                        # , 'VAFIN': ['number', 'person']
                        })
    else:
        match = re.match(r'^form-unk-(\d+).*$', terminal_labeling)
        if match:
            unk_threshold = int(match.group(1))
            term_labelling = grammar.induction.terminal_labeling.FormPosTerminalsUnk(corpus_induce.get_trees(), unk_threshold, pos_filter=["NE", "CARD"])
        else:
            term_labelling = grammar.induction.terminal_labeling.the_terminal_labeling_factory().get_strategy(terminal_labeling)

    if not os.path.isdir(dir):
        os.makedirs(dir)

    # start actual training
    # we use the training corpus until limit for grammar induction (i.e., also the validation section)
    print("Computing baseline id: ")
    baseline_id = grammar_id(corpus_induce, nonterminal_labeling, term_labelling, recursive_partitioning)
    print(baseline_id)
    baseline_path = compute_grammar_name(dir, baseline_id, "baseline")

    if recompileGrammar or not os.path.isfile(baseline_path):
        print("Inducing grammar from corpus")
        (n_trees, baseline_grammar) = d_i.induce_grammar(corpus_induce.get_trees(), nonterminal_labeling, term_labelling.token_label, recursive_partitioning, start)
        print("Induced grammar using", n_trees, ".")
        pickle.dump(baseline_grammar, open(baseline_path, 'wb'))
    else:
        print("Loading grammar from file")
        baseline_grammar = pickle.load(open(baseline_path))

    print("Rules: ", len(baseline_grammar.rules()))

    if parsing:
        parser_ = GFParser_k_best if parser == Coarse_to_fine_parser else parser
        baseline_parser = do_parsing(baseline_grammar, corpus_test, term_labelling, result, baseline_id, parser_, k_best=k_best, minimum_risk=minimum_risk, oracle_parse=oracle_parse, recompile=recompileGrammar, dir=dir, reparse=reparse)

    if True:
        em_trained = pickle.load(open(baseline_path))
        reduct_path = compute_reduct_name(dir, baseline_id, corpus_train)
        if recompileGrammar or not os.path.isfile(reduct_path):
            trace = compute_reducts(em_trained, corpus_train.get_trees(), term_labelling)
            trace.serialize(reduct_path)
        else:
            print("loading trace")
            trace = PySDCPTraceManager(em_trained, term_labelling)
            trace.load_traces_from_file(reduct_path)

        if discr:
            reduct_path_discr = compute_reduct_name(dir, baseline_id, corpus_train, '_discr')
            if recompileGrammar or not os.path.isfile(reduct_path_discr):
                trace_discr = compute_LCFRS_reducts(em_trained, corpus_train.get_trees(), terminal_labelling=term_labelling, nonterminal_map=trace.get_nonterminal_map())
                trace_discr.serialize(reduct_path_discr)
            else:
                print("loading trace discriminative")
                trace_discr = PyLCFRSTraceManager(em_trained, trace.get_nonterminal_map())
                trace_discr.load_traces_from_file(reduct_path_discr)

        # todo refactor EM training, to use the LA version (but without any splits)
        """
        em_trained_path_ = em_trained_path(dir, grammar_id, n_epochs=emEpochs, init=emInit, tie_breaking=emTieBreaking, seed=seed)

        if recompileGrammar or retrain or not os.path.isfile(em_trained_path_):
            emTrainer = PyEMTrainer(trace)
            emTrainer.em_training(em_trained, n_epochs=emEpochs, init=emInit, tie_breaking=emTieBreaking, seed=seed)
            pickle.dump(em_trained, open(em_trained_path_, 'wb'))
        else:
            em_trained = pickle.load(open(em_trained_path_, 'rb'))

        if parsing:
            do_parsing(em_trained, test_limit, ignore_punctuation, term_labelling, recompileGrammar or retrain, [dir, "em_trained_gf_grammar"])
        """

        grammarInfo = PyGrammarInfo(baseline_grammar, trace.get_nonterminal_map())
        storageManager = PyStorageManager()

        builder = PySplitMergeTrainerBuilder(trace, grammarInfo)
        builder.set_em_epochs(emEpochs)
        builder.set_smoothing_factor(rule_smoothing)
        builder.set_split_randomization(splitRandomization, seed + 1)
        if discr:
            builder.set_discriminative_expector(trace_discr, maxScale=maxScaleDiscr, threads=1)
        else:
            builder.set_simple_expector(threads=1)
        if validation:
            if validationMethod is "likelihood":
                reduct_path_validation = compute_reduct_name(dir, baseline_id, corpus_validation)
                if recompileGrammar or not os.path.isfile(reduct_path_validation):
                    validation_trace = compute_reducts(em_trained, corpus_validation.get_trees(), term_labelling)
                    validation_trace.serialize(reduct_path_validation)
                else:
                    print("loading trace validation")
                    validation_trace = PySDCPTraceManager(em_trained, term_labelling)
                    validation_trace.load_traces_from_file(reduct_path_validation)
                builder.set_simple_validator(validation_trace, maxDrops=validationDropIterations, threads=1)
            else:
                validator = build_score_validator(baseline_grammar, grammarInfo, trace.get_nonterminal_map(), storageManager, term_labelling, baseline_parser, corpus_validation, validationMethod)
                builder.set_score_validator(validator, validationDropIterations)
        splitMergeTrainer = builder.set_percent_merger(mergePercentage).build()
        if validation:
            splitMergeTrainer.setMaxDrops(1, mode="smoothing")
            splitMergeTrainer.setEMepochs(1, mode="smoothing")

        sm_info_path = compute_sm_info_path(dir, baseline_id, emEpochs, rule_smoothing, splitRandomization, seed, discr, validation, corpus_validation, emInit)

        if (not recompileGrammar) and (not retrain) and os.path.isfile(sm_info_path):
            print("Loading splits and weights of LA rules")
            latentAnnotation = map(lambda t: build_PyLatentAnnotation(t[0], t[1], t[2], grammarInfo, storageManager)
                                   , pickle.load(open(sm_info_path, 'rb')))
        else:
            # latentAnnotation = [build_PyLatentAnnotation_initial(em_trained, grammarInfo, storageManager)]
            latentAnnotation = [build_PyLatentAnnotation_initial(baseline_grammar, grammarInfo, storageManager)]

        for cycle in range(smCycles + 1):
            if cycle < len(latentAnnotation):
                smGrammar = latentAnnotation[cycle].build_sm_grammar(baseline_grammar
                                                                     , grammarInfo
                                                                     , rule_pruning=rule_pruning
                                                                     # , rule_smoothing=rule_smoothing
                                                                     )
            else:
                # setting the seed to achieve reproducibility in case of continued training
                splitMergeTrainer.reset_random_seed(seed + cycle + 1)
                latentAnnotation.append(splitMergeTrainer.split_merge_cycle(latentAnnotation[-1]))
                pickle.dump(map(lambda la: la.serialize(), latentAnnotation), open(sm_info_path, 'wb'))
                smGrammar = latentAnnotation[cycle].build_sm_grammar(baseline_grammar
                                                                 , grammarInfo
                                                                 , rule_pruning=rule_pruning
                                                                 # , rule_smoothing=rule_smoothing
                                                                    )
            print("Cycle: ", cycle, "Rules: ", len(smGrammar.rules()))
            if parsing:
                grammar_identifier = compute_sm_grammar_id(baseline_id, emEpochs, rule_smoothing, splitRandomization, seed, discr, validation, corpus_validation, emInit, cycle)
                if parser == Coarse_to_fine_parser:
                    opt = { 'latentAnnotation': latentAnnotation[:cycle+1]#[cycle]
                           , 'grammarInfo': grammarInfo
                            , 'nontMap': trace.get_nonterminal_map()}
                    do_parsing(baseline_grammar, corpus_test, term_labelling, result, grammar_identifier, parser,
                               k_best=k_best, minimum_risk=minimum_risk, oracle_parse=oracle_parse,
                               recompile=recompileGrammar, dir=dir, reparse=reparse, opt=opt)
                else:
                    do_parsing(smGrammar, corpus_test, term_labelling, result, grammar_identifier, parser, k_best=k_best, minimum_risk=minimum_risk, oracle_parse=oracle_parse, recompile=recompileGrammar, dir=dir, reparse=reparse)


def do_parsing(grammar, test_corpus, term_labelling, result, grammar_identifier, parser_type, k_best, minimum_risk=False, oracle_parse=False, recompile=True, reparse=False, dir=None, opt=None):
    tree_yield = term_labelling.prepare_parser_input

    result_path = result(grammar_identifier)
    minimum_risk_path = result(grammar_identifier, 'min_risk')
    oracle_parse_path = result(grammar_identifier, 'oracle_file')

    total_time = 0.0

    preprocess_path = [os.path.join(dir, grammar_identifier), "gf_grammar"]
    # print(preprocess_path)
    load_preprocess = preprocess_path
    if parser_type not in [GFParser, GFParser_k_best, Coarse_to_fine_parser] \
            or recompile \
            or (not os.path.isfile(GFParser.resolve_path(preprocess_path))):
        load_preprocess=None
    if parser_type in [GFParser, GFParser_k_best, Coarse_to_fine_parser] \
            and not os.path.isdir(os.path.join(dir, grammar_identifier)):
        os.makedirs(os.path.join(dir, grammar_identifier))

    if parser_type == GFParser_k_best:
        parser = GFParser_k_best(grammar, save_preprocessing=preprocess_path, load_preprocessing=load_preprocess, k=k_best)
    elif parser_type == Coarse_to_fine_parser:
        parser = Coarse_to_fine_parser(grammar, base_parser_type=GFParser_k_best, la=opt["latentAnnotation"], grammarInfo=opt["grammarInfo"], nontMap=opt["nontMap"], save_preprocessing=preprocess_path, load_preprocessing=load_preprocess, k=k_best)
    else:
        parser = parser_type(grammar, save_preprocess=preprocess_path, load_preprocess=load_preprocess)

    if recompile or reparse or \
            not os.path.isfile(result_path) \
            or (minimum_risk and not os.path.isfile(minimum_risk_path)) \
            or (oracle_parse and not os.path.isfile(oracle_parse_path)):

        result_dirs = map(lambda path: os.path.split(path)[0], [result_path, minimum_risk_path, oracle_parse_path])
        for result_dir in result_dirs:
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)

        with open(result_path, 'w') as result_file, \
                open(minimum_risk_path, 'w') as minimum_risk_file, \
                open(oracle_parse_path, 'w') as oracle_parse_file:
            failures = 0
            for tree in test_corpus.get_trees():
                time_stamp = time.clock()

                parser.set_input(tree_yield(tree.token_yield()))
                parser.parse()
                # if not parser.recognized():
                #     parser = parser_type(grammar_second, tree_yield(tree.token_yield()))
                # if not parser.recognized():
                #     parser = parser_type(grammar_tern, tree_yield(tree.token_yield()))
                time_stamp = time.clock() - time_stamp
                total_time += time_stamp

                cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                for token in cleaned_tokens:
                    token.set_edge_label('_')

                h_tree = HybridTree(tree.sent_label())

                if parser_type in [GFParser_k_best, Coarse_to_fine_parser] and parser.recognized():
                    if minimum_risk or oracle_parse:
                        h_trees = []
                        weights = []

                        for weight, der in parser.k_best_derivation_trees():

                            dcp = DCP_evaluator(der).getEvaluation()
                            h_tree = HybridTree()
                            cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                            dcp_to_hybridtree(h_tree, dcp, cleaned_tokens, False, construct_conll_token)

                            h_trees.append(h_tree)
                            weights.append(weight)

                        if minimum_risk:
                            h_tree_min_risk = compute_minimum_risk_tree(h_trees, weights)
                        if oracle_parse:
                            h_tree_oracle = compute_oracle_tree(h_trees, tree)

                    der_to_tree = lambda der: dcp_to_hybridtree(HybridTree(), DCP_evaluator(der).getEvaluation(),
                                                                copy.deepcopy(tree.full_token_yield()), False,
                                                                construct_conll_token)
                    # h_tree = parser.best_trees(der_to_tree)[0][0]
                    h_tree = HybridTree(tree.sent_label())
                    h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, copy.deepcopy(tree.full_token_yield()), ignore_punctuation, construct_conll_token)
                elif parser_type == CFGParser \
                        or parser_type == GFParser \
                        or parser_type == LeftBranchingFSTParser \
                        or parser_type == RightBranchingFSTParser:
                    h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, ignore_punctuation,
                                                                  construct_conll_token)
                else:
                    h_tree = None

                if h_tree:
                    result_file.write(tree_to_conll_str(h_tree))
                    result_file.write('\n\n')
                    if minimum_risk and parser_type in [GFParser_k_best, Coarse_to_fine_parser]:
                        minimum_risk_file.write(tree_to_conll_str(h_tree_min_risk))
                        minimum_risk_file.write('\n\n')
                    if oracle_parse and parser_type in [GFParser_k_best, Coarse_to_fine_parser]:
                        oracle_parse_file.write(tree_to_conll_str(h_tree_oracle))
                        oracle_parse_file.write('\n\n')
                else:
                    failures += 1
                    forms = [token.form() for token in tree.full_token_yield()]
                    poss = [token.pos() for token in tree.full_token_yield()]
                    fall_back = tree_to_conll_str(fall_back_left_branching(forms, poss))
                    files = [result_file]
                    if minimum_risk:
                        files.append(minimum_risk_file)
                    if oracle_parse:
                        files.append(oracle_parse_file)
                    for file in files:
                        file.write(fall_back)
                        file.write('\n\n')

                parser.clear()

        print("parse failures", failures)
        print("parse time", total_time)

    if parser_type == GFParser_k_best:
        print("best parse results")
    else:
        print("viterbi parse results")
    eval_pl_call(test_corpus._path, result_path)
    if oracle_parse:
        print("\noracle parse results")
        eval_pl_call(test_corpus._path, oracle_parse_path)
    if minimum_risk:
        print("\nminimum risk results")
        eval_pl_call(test_corpus._path, minimum_risk_path)

    return parser


def eval_pl_call(test_path, result_path):
    print("eval.pl", "no punctuation")
    p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test_path, "-s", result_path, "-q"])
    p.communicate()
    print("eval.pl", "punctation")
    p = subprocess.Popen(
        ["perl", "../util/eval.pl", "-g", test_path, "-s", result_path, "-q", "-p"])
    p.communicate()


def length_limit(trees, max_length=50):
    for tree in trees:
        if len(tree.full_token_yield()) <= max_length:
            yield tree


def build_score_validator(baseline_grammar, grammarInfo, nont_map, storageManager, term_labelling, parser, corpus_validation, validationMethod):
    validator = PyCandidateScoreValidator(grammarInfo, storageManager, validationMethod)

    # parser = GFParser(baseline_grammar)
    tree_count = 0
    der_count = 0
    for gold_tree in corpus_validation.get_trees():
        tree_count += 1
        parser.set_input(term_labelling.prepare_parser_input(gold_tree.token_yield()))
        parser.parse()
        derivations = map(lambda x: x[1], parser.k_best_derivation_trees())
        manager = PyDerivationManager(baseline_grammar, nont_map)
        manager.convert_derivations_to_hypergraphs(derivations)
        scores = []

        gold_labels = {}
        gold_heads = {}

        for position, id in enumerate(gold_tree.id_yield()):
            parent_id = gold_tree.parent(id)
            gold_labels[position] = gold_tree.node_token(id).deprel()
            if parent_id is None:
                assert id in gold_tree.root
                gold_heads[position] = 0
            else:
                gold_heads[position] = gold_tree.id_yield().index(parent_id) + 1

        derivations = parser.k_best_derivation_trees()
        for _, der in derivations:
            der_count += 1
            h_tree = HybridTree()
            cleaned_tokens = copy.deepcopy(gold_tree.full_token_yield())
            dcp = DCP_evaluator(der).getEvaluation()
            dcp_to_hybridtree(h_tree, dcp, cleaned_tokens, False, construct_conll_token)

            las, uas, lac = 0, 0, 0
            for position, id in enumerate(h_tree.id_yield()):
                parent_id = h_tree.parent(id)
                if parent_id is None:
                    assert id in h_tree.root
                    head = 0
                else:
                    head = h_tree.id_yield().index(parent_id) + 1
                label = h_tree.node_token(id).deprel()

                if gold_heads[position] == head:
                    uas += 1
                if gold_labels[position] == label:
                    lac += 1
                if gold_heads[position] == head and gold_labels[position] == label:
                    las += 1

            if validationMethod == "LAS":
                scores.append(las)
            elif validationMethod == "UAS":
                scores.append(uas)
            elif validationMethod == "LAC":
                scores.append(lac)

        max_score = len(gold_tree.id_yield())
        validator.add_scored_candidates(manager, scores, max_score)
        print(tree_count, max_score, scores)
        parser.clear()

    print("trees used for validation ", tree_count, "with", der_count * 1.0 / tree_count, "derivations on average")

    return validator


if __name__ == '__main__':
    plac.call(main)
