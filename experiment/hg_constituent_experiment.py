from __future__ import print_function
from ast import literal_eval
import copy
from collections import defaultdict
from functools import lru_cache
import os
import pickle
import sys
import subprocess
import tempfile
import itertools
from hybridtree.general_hybrid_tree import HybridTree
from parser.discodop_parser.parser import DiscodopKbestParser
try:
    from parser.gf_parser.gf_interface import GFParser_k_best
except ImportError:
    print("The Grammatical Framework is not installed properly – the GFParser is unavailable.")
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
from parser.sDCPevaluation.evaluator import DCP_evaluator, dcp_to_hybridtree
from parser.trace_manager.sm_trainer import build_PyLatentAnnotation
from parser.lcfrs_la import construct_fine_grammar
import plac
import json
from grammar.induction.terminal_labeling import TerminalLabeling, deserialize_labeling, StanfordUNKing
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from constituent.induction import fringe_extract_lcfrs, token_to_features
from constituent.construct_morph_annotation import build_nont_splits_dict, pos_cat_and_lex_in_unary, \
    extract_feat
from constituent.discodop_adapter import TreeComparator as DiscoDopScorer
from constituent.dummy_tree import flat_dummy_constituent_tree
from constituent.parse_accuracy import ParseAccuracyPenalizeFailures
import corpora.tiger_parse as tp
import corpora.negra_parse as np
import corpora.tagged_parse as tagged_parse
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.monadic_tokens import construct_constituent_token
from experiment.base_experiment import ScoringExperiment
from experiment.resources import TRAINING, VALIDATION, TESTING, TESTING_INPUT, RESULT, CorpusFile, ScorerResource
from experiment.split_merge_experiment import SplitMergeExperiment
from experiment.constituent_experiment_helpers import *


MULTI_OBJECTIVES = "multi-objectives"
MULTI_OBJECTIVES_INDEPENDENT = "multi-objectives-independent"
NO_PARSING = "no-parsing"
BASE_GRAMMAR = "base-grammar" # use base grammar for parsing (no annotations LA)
MAX_RULE_PRODUCT_ONLY = "max-rule-product-only"
TEST_SECOND_HALF = False

MAX_SENTENCE_LENGTH = 5000
NEGRA = "NEGRA"


class FeatureFunction:
    def __init__(self):
        self.function = pos_cat_and_lex_in_unary
        self.default_args = {'hmarkov': 1}

    def __call__(self, *args):
        return self.function(*args, **self.default_args)

    def __str__(self):
        __str = "Feature Function {"
        __str += "func: " + str(self.function)
        __str += "kwargs: " + str(self.default_args)
        __str += "}"
        return __str


class InductionSettings:
    """
    Holds settings for a hybrid grammar parsing experiment.
    """
    def __init__(self):
        self.recursive_partitioning = None
        self.terminal_labeling = None
        self.isolate_pos = False
        self.naming_scheme = 'child'
        self.edge_labels = True
        self.disconnect_punctuation = True
        self.normalize = False
        self.feature_la = False
        self.feat_function = FeatureFunction()
        self.discodop_binarization_params = ["--headrules=util/negra.headrules",
                                             "--binarize"]

    def __str__(self):
        __str = "Induction Settings {\n"
        for key in self.__dict__:
            if not key.startswith("__") and key not in []:
                __str += "\t" + key + ": " + str(self.__dict__[key]) + "\n"
        return __str + "}"


class ConstituentScorer(ScorerResource):
    """
    A resource to which parsing results can be written.
    Computes LF1 score based on an in house implementation of the PARSEVAL metric.
    """
    def __init__(self):
        super(ConstituentScorer, self).__init__()
        self.scorer = ParseAccuracyPenalizeFailures()

    def score(self, system, gold, secondaries=None):
        self.scorer.add_accuracy(system.labelled_spans(), gold.labelled_spans())

    def failure(self, gold):
        self.scorer.add_failure(gold.labelled_spans())


class ScorerAndWriter(ConstituentScorer, CorpusFile):
    """
    A resource to which parsing results can be written.
    Computes LF1 score (inhouse implementation) and writes resulting parse tree to a file.
    """
    def __init__(self, experiment, path=None, directory=None, logger=None, secondary_scores=0):
        ConstituentScorer.__init__(self)
        _, path = tempfile.mkstemp(dir=directory) if path is None else path
        CorpusFile.__init__(self, path=path, directory=directory, logger=logger)
        self.experiment = experiment
        self.reference = CorpusFile(directory=directory, logger=logger)
        self.logger = logger if logger is not None else sys.stdout
        self.secondaries = [CorpusFile(directory=directory, logger=logger) for _ in range(secondary_scores)]

    def init(self):
        CorpusFile.init(self)
        self.reference.init()
        for sec in self.secondaries:
            sec.init()

    def finalize(self):
        CorpusFile.finalize(self)
        self.reference.finalize()
        print('Wrote results to', self.path, file=self.logger)
        print('Wrote reference to', self.reference.path, file=self.logger)
        for i, sec in enumerate(self.secondaries):
            sec.finalize()
            print('Wrote sec %d to ' % i, sec.path, file=self.logger)

    def score(self, system, gold, secondaries=None):
        ConstituentScorer.score(self, system, gold)
        self.file.writelines(self.experiment.serialize(system))
        self.reference.file.writelines(self.experiment.serialize(gold))
        if secondaries:
            for system_sec, corpus in zip(secondaries, self.secondaries):
                corpus.file.writelines(self.experiment.serialize(system_sec))

    def failure(self, gold):
        ConstituentScorer.failure(self, gold)
        sentence = self.experiment.obtain_sentence(gold)
        label = self.experiment.obtain_label(gold)
        fallback = self.experiment.compute_fallback(sentence, label)
        self.file.writelines(self.experiment.serialize(fallback))
        self.reference.file.writelines(self.experiment.serialize(gold))
        for sec in self.secondaries:
            sec.file.writelines(self.experiment.serialize(fallback))

    def __str__(self):
        return CorpusFile.__str__(self)


class ConstituentExperiment(ScoringExperiment):
    """
    Holds state and methods of a constituent parsing experiment.
    """
    def __init__(self, induction_settings, directory=None, filters=None):
        ScoringExperiment.__init__(self, directory=directory, filters=filters)
        self.induction_settings = induction_settings
        self.resources[RESULT] = ScorerAndWriter(self, directory=self.directory, logger=self.logger)
        self.serialization_type = NEGRA
        self.use_output_counter = False
        self.output_counter = 0
        self.strip_vroot = False
        self.terminal_labeling = None
        self.eval_postprocess_options = None

        self.discodop_scorer = DiscoDopScorer('util/proper.prm')
        self.max_score = 100.0

        self.backoff = False
        self.backoff_factor = 10.0

    def obtain_sentence(self, obj):
        if isinstance(obj, HybridTree):
            sentence = obj.full_yield(), obj.id_yield(), \
                       obj.full_token_yield(), obj.token_yield()
            return sentence
        elif isinstance(obj, list):
            return [i for i in range(len(obj))], [i for i in range(len(obj))], obj, obj
        else:
            raise ValueError("Unsupported obj type", type(obj), "instance", obj)

    def obtain_label(self, hybrid_tree):
        return hybrid_tree.sent_label()

    def compute_fallback(self, sentence, label=None):
        full_yield, id_yield, full_token_yield, token_yield = sentence
        return flat_dummy_constituent_tree(token_yield, full_token_yield, 'NP', 'S', label)

    def read_stage_file(self):
        ScoringExperiment.read_stage_file(self)
        if "terminal_labeling" in self.stage_dict:
            terminal_labeling_path = self.stage_dict["terminal_labeling"]
            with open(terminal_labeling_path, "r") as tlf:
                self.terminal_labeling = deserialize_labeling(json.load(tlf))
                self.induction_settings.terminal_labeling = self.terminal_labeling

    def read_corpus(self, resource):
        if resource.type == "TIGERXML":
            return self.read_corpus_tigerxml(resource)
        elif resource.type == "EXPORT":
            return self.read_corpus_export(resource)
        elif resource.type == "WORD/POS":
            return self.read_corpus_tagged(resource)
        else:
            raise ValueError("Unsupport resource type " + resource.type)

    def read_corpus_tigerxml(self, resource):
        """
        :type resource: CorpusFile
        :return: corpus of constituent trees
        """
        path = resource.path
        prefix = 's'
        if self.induction_settings.normalize:
            path = self.normalize_corpus(path, src='tigerxml', dest='tigerxml', renumber=False)
            prefix = ''

        if resource.filter is None:
            def sentence_filter(_):
                return True
        else:
            sentence_filter = resource.filter

        return tp.sentence_names_to_hybridtrees(
            [prefix + str(i) for i in range(resource.start, resource.end + 1)
             if i not in resource.exclude and sentence_filter(i)]
            , path
            , hold=False
            , disconnect_punctuation=self.induction_settings.disconnect_punctuation)

    def read_corpus_export(self, resource, mode="STANDARD", skip_normalization=False):
        """
        :type resource: CorpusFile
        :param mode: either STANDARD or DISCO-DOP (handles variation in NEGRA format)
        :type mode: str
        :param skip_normalization: If normalization is skipped even if set in induction settings.
        :type skip_normalization: bool
        :return: corpus of constituent trees
        """
        if resource.filter is None:
            def sentence_filter(_):
                return True
        else:
            sentence_filter = resource.filter
        path = resource.path
        if not skip_normalization and self.induction_settings.normalize:
            path = self.normalize_corpus(path, src='export', dest='export', renumber=False)
        # encoding = "iso-8859-1"
        encoding = "utf-8"
        return np.sentence_names_to_hybridtrees(
            {str(i) for i in range(resource.start, resource.end + 1)
             if i not in resource.exclude and sentence_filter(i)},
            path,
            enc=encoding, disconnect_punctuation=self.induction_settings.disconnect_punctuation, add_vroot=True,
            mode=mode)

    def run_discodop_binarization(self):
        """
        :rtype: None
        Binarize the training corpus using discodop. The resulting corpus is saved in resources_data
        under the key disco_binarized_corus.
        """
        if self.resources_data.get('disco_binarized_corpus', None) is not None:
            return
        train_resource = self.resources[TRAINING]
        if self.induction_settings.normalize:
            train_normalized = self.normalize_corpus(train_resource.path,
                                                     src=train_resource.type.lower(),
                                                     dest='export',
                                                     renumber=False)
        else:
            train_normalized = train_resource.path

        _, second_stage = tempfile.mkstemp(suffix=".export", dir=self.directory)

        subprocess.call(["discodop", "treetransforms"]
                        + self.induction_settings.discodop_binarization_params
                        + ["--inputfmt=export", "--outputfmt=export",
                           train_normalized, second_stage])

        disco_resource = CorpusFile(path=second_stage,
                                    start=train_resource.start,
                                    end=train_resource.end,
                                    limit=train_resource.limit,
                                    filter=train_resource.filter,
                                    exclude=train_resource.exclude,
                                    type=train_resource.type
                                   )

        self.resources_data['disco_binarized_corpus'] \
            = self.read_corpus_export(disco_resource, mode="DISCO-DOP", skip_normalization=True)

    def read_corpus_tagged(self, resource):
        return itertools.islice(tagged_parse.parse_tagged_sentences(resource.path), resource.start, resource.limit)

    def parsing_preprocess(self, obj):
        if isinstance(obj, HybridTree):
            if True or self.strip_vroot:
                obj.strip_vroot()
            parser_input = self.terminal_labeling.prepare_parser_input(obj.token_yield())
            # print(parser_input)
            return parser_input
        else:
            return self.terminal_labeling.prepare_parser_input(obj)

    def parsing_postprocess(self, sentence, derivation, label=None):
        full_yield, id_yield, full_token_yield, token_yield = sentence

        dcp_tree = ConstituentTree(label)
        punctuation_positions = [i + 1 for i, idx in enumerate(full_yield)
                                 if idx not in id_yield]

        cleaned_tokens = copy.deepcopy(full_token_yield)
        dcp = DCP_evaluator(derivation).getEvaluation()
        dcp_to_hybridtree(dcp_tree, dcp, cleaned_tokens, False, construct_constituent_token,
                          punct_positions=punctuation_positions)

        if True or self.strip_vroot:
            dcp_tree.strip_vroot()

        return dcp_tree

    def preprocess_before_induction(self, obj):
        if self.strip_vroot:
            obj.strip_vroot()
        return obj

    @lru_cache(maxsize=500)
    def normalize_corpus(self, path, src='export', dest='export', renumber=True, disco_options=None):
        _, first_stage = tempfile.mkstemp(suffix=".export", dir=self.directory)
        subprocess.call(["treetools", "transform", path, first_stage, "--trans", "root_attach",
                         "--src-format", src, "--dest-format", "export"])
        _, second_stage = tempfile.mkstemp(suffix=".export", dir=self.directory)
        second_call = ["discodop", "treetransforms"]
        if renumber:
            second_call.append("--renumber")
        if disco_options:
            second_call += list(disco_options)
        subprocess.call(second_call + ["--punct=move", first_stage, second_stage,
                                       "--inputfmt=export", "--outputfmt=export"])
        if dest == 'export':
            return second_stage
        elif dest == 'tigerxml':
            _, third_stage = tempfile.mkstemp(suffix=".xml", dir=self.directory)
            subprocess.call(["treetools", "transform", second_stage, third_stage,
                             "--src-format", "export", "--dest-format", dest])
            return third_stage
        raise ValueError("Unsupported dest format", dest)

    def evaluate(self, result_resource, gold_resource):
        accuracy = result_resource.scorer
        print('', file=self.logger)
        # print('Parsed:', n)
        if accuracy.n() > 0:
            print('Recall:   ', accuracy.recall(), file=self.logger)
            print('Precision:', accuracy.precision(), file=self.logger)
            print('F-measure:', accuracy.fmeasure(), file=self.logger)
            print('Parse failures:', accuracy.n_failures(), file=self.logger)
        else:
            print('No successful parsing', file=self.logger)
        # print('time:', end_at - start_at)
        print('')

        print('normalize results with treetools and discodop', file=self.logger)

        ref_rn = self.normalize_corpus(result_resource.reference.path, disco_options=self.eval_postprocess_options)
        sys_rn = self.normalize_corpus(result_resource.path, disco_options=self.eval_postprocess_options)
        sys_secs = [self.normalize_corpus(sec.path, disco_options=self.eval_postprocess_options)
                    for sec in result_resource.secondaries]

        prm = "util/proper.prm"

        def run_eval(sys_path, mode):
            print(mode)
            print('running discodop evaluation on gold:', ref_rn, ' and sys:', sys_path,
                  "with", os.path.split(prm)[1], file=self.logger)
            output = subprocess.Popen(["discodop", "eval", ref_rn, sys_path, prm],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT).communicate()
            print(str(output[0], encoding='utf-8'), file=self.logger)

        run_eval(sys_rn, "DEFAULT")

        for i, sec in enumerate(sys_secs):
            run_eval(sec, self.parser.secondaries[i])

    @staticmethod
    def __obtain_labelled_spans(obj):
        spans = obj.labelled_spans()
        spans = map(tuple, spans)
        spans = set(spans)
        return spans

    def score_object(self, obj, gold):
        # _, _, lf1 = self.precision_recall_f1(self.__obtain_labelled_spans(gold), self.__obtain_labelled_spans(obj))
        lf1 = self.discodop_scorer.compare_hybridtrees(gold, obj)
        return lf1

    def serialize(self, obj):
        if self.serialization_type == NEGRA:
            if self.use_output_counter:
                self.output_counter += 1
                number = self.output_counter
            else:
                label = self.obtain_label(obj)
                if label.startswith('s'):
                    number = int(label[1:])
                else:
                    number = int(label)
            return np.serialize_hybridtrees_to_negra([obj], number, MAX_SENTENCE_LENGTH)
        raise ValueError("Unsupported serialization type", self.serialization_type)

    def print_config(self, file=None):
        if file is None:
            file = self.logger
        ScoringExperiment.print_config(self, file=file)
        print(self.induction_settings, file=file)
        print("k-best", self.k_best, file=file)
        print("Serialization type", self.serialization_type, file=file)
        print("Output counter", self.use_output_counter, "start", self.output_counter, file=file)
        print("VROOT stripping", self.strip_vroot, file=file)
        print("Max score", self.max_score, file=file)
        print("Backoff", self.backoff, file=file)
        print("Backoff-factor", self.backoff_factor, file=file)

    def set_terminal_labeling(self, terminal_labeling):
        """
        :type terminal_labeling: TerminalLabeling
        Sets a terminal labeling, serializes it, and adds it to the stage file.
        """
        self.terminal_labeling = terminal_labeling
        self.induction_settings.terminal_labeling = terminal_labeling
        _, path = tempfile.mkstemp(dir=self.directory, suffix=".lexicon")
        with open(path, 'w') as fd:
            self.stage_dict["terminal_labeling"] = path
            json.dump(self.terminal_labeling.serialize(), fd)
        self.write_stage_file()


class ConstituentSMExperiment(ConstituentExperiment, SplitMergeExperiment):
    """
    Extends constituent parsing experiment by providing methods for dealing with
    latent annotation extensions of LCFRS or LCFRS/sDCP hybrid grammars.
    """
    def __init__(self, induction_settings, directory=None):
        """
        :type induction_settings: InductionSettings
        """
        ConstituentExperiment.__init__(self, induction_settings, directory=directory)
        SplitMergeExperiment.__init__(self)
        self.k_best = 50
        self.rule_smooth_list = None
        if self.induction_settings.feature_la:
            self.feature_log = defaultdict(lambda: 0)

    def initialize_parser(self):
        if "disco-dop" in self.parsing_mode:
            self.parser = DiscodopKbestParser(grammar=self.base_grammar,
                                              k=self.k_best,
                                              beam_beta=self.disco_dop_params["beam_beta"],
                                              beam_delta=self.disco_dop_params["beam_delta"],
                                              pruning_k=self.disco_dop_params["pruning_k"],
                                              cfg_ctf=self.disco_dop_params["cfg_ctf"])
        else:
            self.parser = GFParser_k_best(grammar=self.base_grammar, k=self.k_best,
                                          save_preprocessing=(self.directory, "gfgrammar"))

    def read_stage_file(self):
        ConstituentExperiment.read_stage_file(self)

        if "training_reducts" in self.stage_dict:
            self.organizer.training_reducts = PySDCPTraceManager(self.base_grammar, self.terminal_labeling)
            self.organizer.training_reducts.load_traces_from_file(
                bytes(self.stage_dict["training_reducts"], encoding="utf-8"))

        if "validation_reducts" in self.stage_dict:
            self.organizer.validation_reducts = PySDCPTraceManager(self.base_grammar, self.terminal_labeling)
            self.organizer.validation_reducts.load_traces_from_file(
                bytes(self.stage_dict["validation_reducts"], encoding="utf-8"))

        if "rule_smooth_list" in self.stage_dict:
            with open(self.stage_dict["rule_smooth_list"]) as file:
                self.rule_smooth_list = pickle.load(file)

        SplitMergeExperiment.read_stage_file(self)

    def __grammar_induction(self, tree, part, features):
        return fringe_extract_lcfrs(tree, part, naming=self.induction_settings.naming_scheme,
                                    term_labeling=self.induction_settings.terminal_labeling,
                                    isolate_pos=self.induction_settings.isolate_pos,
                                    feature_logging=features)

    def additional_induction_params(self, obj):
        if 'guided_binarization' in self.induction_settings.recursive_partitioning.__name__:
            return {'rec_part_params':
                    {'reference_tree': self.resources_data['binarized_corpus_dict'].get(obj.sent_label())}}
        else:
            return {}

    def induction_preparation(self):
        if 'guided_binarization' in self.induction_settings.recursive_partitioning.__name__:
            if self.resources_data.get('binarized_corpus_dict', None) is None:
                self.run_discodop_binarization()
                self.resources_data['binarized_corpus_dict'] = {
                    tree.sent_label(): tree for tree in self.resources_data['disco_binarized_corpus']
                }

    def induce_from(self, obj, **kwargs):
        if not obj.complete() or obj.empty_fringe():
            return None, None

        rec_part_params = kwargs['rec_part_params'] if 'rec_part_params' in kwargs else {}
        part = self.induction_settings.recursive_partitioning(obj, **rec_part_params)

        features = defaultdict(lambda: 0) if self.induction_settings.feature_la else None

        if not self.induction_settings.edge_labels:
            obj.reset_edge_labels('--')

        tree_grammar = self.__grammar_induction(obj, part, features)

        if self.backoff:
            self.terminal_labeling.backoff_mode = True

            features_backoff = defaultdict(lambda: 0) if self.induction_settings.feature_la else None
            tree_grammar_backoff = self.__grammar_induction(obj, part, features=features_backoff)
            tree_grammar.add_gram(tree_grammar_backoff,
                                  feature_logging=(features, features_backoff) if features_backoff else None)

            self.terminal_labeling.backoff_mode = False

        if False and len(obj.token_yield()) == 1:
            print(obj, map(str, obj.token_yield()), file=self.logger)
            print(tree_grammar, file=self.logger)

        return tree_grammar, features

    def print_config(self, file=None):
        if file is None:
            file = self.logger
        ConstituentExperiment.print_config(self, file=file)
        SplitMergeExperiment.print_config(self, file=file)

    def compute_reducts(self, resource):
        corpus = self.read_corpus(resource)
        if self.strip_vroot:
            for tree in corpus:
                tree.strip_vroot()
        if not self.induction_settings.edge_labels:
            for tree in corpus:
                tree.reset_edge_labels('--')
            
        parser = self.organizer.training_reducts.get_parser() if self.organizer.training_reducts is not None else None
        nonterminal_map = self.organizer.nonterminal_map
        frequency = self.backoff_factor if self.backoff else 1.0
        trace = compute_reducts(self.base_grammar, corpus, self.induction_settings.terminal_labeling,
                                parser=parser, nont_map=nonterminal_map, frequency=frequency)
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            trace.compute_reducts(corpus, frequency=1.0)
            self.terminal_labeling.backoff_mode = False
        return trace

    def create_initial_la(self):
        if self.induction_settings.feature_la:
            print("building initial LA from features", file=self.logger)
            nonterminal_splits, rootWeights, ruleWeights, split_id \
                = build_nont_splits_dict(self.base_grammar,
                                         self.feature_log,
                                         self.organizer.nonterminal_map,
                                         feat_function=self.induction_settings.feat_function)
            print("number of nonterminals:", len(nonterminal_splits), file=self.logger)
            print("total splits", sum(nonterminal_splits), file=self.logger)
            max_splits = max(nonterminal_splits)
            max_splits_index = nonterminal_splits.index(max_splits)
            max_splits_nont = self.organizer.nonterminal_map.index_object(max_splits_index)
            print("max. nonterminal splits", max_splits, "at index ", max_splits_index,
                  "i.e.,", max_splits_nont, file=self.logger)
            for key in split_id[max_splits_nont]:
                print(key, file=self.logger)
            print("splits for NE/1", file=self.logger)
            for key in split_id["NE/1"]:
                print(key, file=self.logger)
            for rule in self.base_grammar.lhs_nont_to_rules("NE/1"):
                print(rule, ruleWeights[rule.get_idx()], file=self.logger)
            print("number of rules", len(ruleWeights), file=self.logger)
            print("total split rules", sum(map(len, ruleWeights)), file=self.logger)
            print("number of split rules with 0 prob.",
                  sum(map(sum, map(lambda xs: map(lambda x: 1 if x == 0.0 else 0, xs), ruleWeights))),
                  file=self.logger)

            la = build_PyLatentAnnotation(nonterminal_splits, rootWeights, ruleWeights, self.organizer.grammarInfo,
                                          self.organizer.storageManager)
            la.add_random_noise(seed=self.organizer.seed)
            self.split_id = split_id
            return la
        else:
            return super(ConstituentSMExperiment, self).create_initial_la()

    def do_em_training(self):
        super(ConstituentSMExperiment, self).do_em_training()
        if self.induction_settings.feature_la:
            self.patch_initial_grammar()

    def custom_sm_options(self, builder):
        if self.rule_smooth_list is not None:
            builder.set_count_smoothing(self.rule_smooth_list, 0.5)
        else:
            SplitMergeExperiment.custom_sm_options(self, builder)

    def postprocess_grammar(self, grammar):
        if isinstance(self.terminal_labeling, StanfordUNKing):
            self.add_smoothed_lex_rules(grammar)
            self.terminal_labeling.test_mode = True
        super(ConstituentExperiment, self).postprocess_grammar(grammar)

    def add_smoothed_lex_rules(self, gr):
        from grammar.lcfrs import LCFRS_lhs
        rules = self.terminal_labeling.create_smoothed_rules()
        new_rules = {}

        for rule in gr.rules():
            if rule.rhs() == []:
                assert len(rule.dcp()) == 1
                dcp = rule.dcp()[0]
                assert len(dcp.rhs()) == 1
                term = dcp.rhs()[0]
                head = term.head()
                pos = head.pos()

                for tag, form in rules:
                    if tag == pos:
                        lhs = LCFRS_lhs(rule.lhs().nont())
                        lhs.add_arg([form])
                        new_rules[lhs, dcp] = rules[tag, form]

        for lhs, dcp in new_rules:
            gr.add_rule(lhs, [], new_rules[(lhs, dcp)], [dcp])

    def patch_initial_grammar(self):
        print("Merging feature splits with SCC merger.", file=self.logger)
        merged_la = self.organizer.emTrainer.merge(self.organizer.latent_annotations[0])
        if False:
            self.organizer.latent_annotations[0] = merged_la
            self.organizer.merge_sources[0] = self.organizer.emTrainer.get_current_merge_sources()
            print(self.organizer.merge_sources[0], file=self.logger)

        else:
            splits, _, _ = merged_la.serialize()
            merge_sources = self.organizer.emTrainer.get_current_merge_sources()

            lookup = self.print_funky_listing(merge_sources)

            fine_grammar_merge_sources = []
            for nont_idx in range(0, self.organizer.nonterminal_map.get_counter()):
                nont = self.organizer.nonterminal_map.index_object(nont_idx)
                if any([rule.rhs() == [] for rule in self.base_grammar.lhs_nont_to_rules(nont)]):
                    fine_grammar_merge_sources.append([[split] for split in range(0, splits[nont_idx])])
                else:
                    fine_grammar_merge_sources.append([[split for split in range(0, splits[nont_idx])]])

            print("Projecting to fine grammar LA", file=self.logger)
            fine_grammar__la = merged_la.project_annotation_by_merging(self.organizer.grammarInfo,
                                                                       fine_grammar_merge_sources)

            def arg_transform(arg, la):
                arg_mod = []
                for elem in arg:
                    if isinstance(elem, str):
                        arg_mod.append(elem + "-group-" + str(la[0]))
                    else:
                        arg_mod.append(elem)
                return arg_mod

            def smooth_transform(arg):
                arg_mod = []
                for elem in arg:
                    if isinstance(elem, str):
                        try:
                            term = literal_eval(elem)
                            if isinstance(term, tuple):
                                pos = dict(term[0]).get("pos", "UNK")
                                arg_mod.append(pos)
                                # print(term, pos, file=self.logger)
                            else:
                                arg_mod.append(elem)
                        except ValueError:
                            arg_mod.append(elem)
                    else:
                        arg_mod.append(elem)
                return arg_mod

            def id_arg(arg, la):
                return arg

            print("Constructing fine grammar", file=self.logger)
            (grammar_fine, grammar_fine_LA_full, grammar_fine_info,
             grammar_fine_nonterminal_map, nont_translation, smooth_rules) \
                = construct_fine_grammar(fine_grammar__la,
                                         self.base_grammar,
                                         self.organizer.grammarInfo,
                                         id_arg,
                                         merged_la,
                                         smooth_transform=smooth_transform)

            self.rule_smooth_list = smooth_rules
            _, path = tempfile.mkstemp(".rule_smooth_list.pkl", dir=self.directory)
            with open(path, 'wb') as file:
                pickle.dump(smooth_rules, file)
                self.stage_dict["rule_smooth_list"] = path

            grammar_fine.make_proper()
            grammar_fine_LA_full.make_proper()
            print(grammar_fine_LA_full.is_proper(), file=self.logger)
            nonterminal_splits, root_weights, rule_weights = grammar_fine_LA_full.serialize()

            # for rule in grammar_fine.rules():
            #     print(rule, rule_weights[rule.get_idx()])
            print("number of nonterminals:", len(nonterminal_splits), file=self.logger)
            print("total splits", sum(nonterminal_splits), file=self.logger)
            print("number of rules", len(rule_weights), file=self.logger)
            print("total split rules", sum(map(len, rule_weights)), file=self.logger)
            print("number of split rules with 0 prob.",
                  sum(map(sum, map(lambda xs: map(lambda x: 1 if x == 0.0 else 0, xs), rule_weights))),
                  file=self.logger)
            # self.base_grammar_backup = self.base_grammar
            self.stage_dict["backup_grammar"] = self.stage_dict["base_grammar"]
            self.base_grammar = grammar_fine
            _, path = tempfile.mkstemp(suffix="basegram.pkl", dir=self.directory)
            with open(path, 'wb') as file:
                pickle.dump(self.base_grammar, file)
                self.stage_dict["base_grammar"] = path

            self.organizer.grammarInfo = grammar_fine_info
            self.organizer.nonterminal_map = grammar_fine_nonterminal_map

            self.organizer.last_sm_cycle = 0
            if True:
                self.organizer.latent_annotations[0] = grammar_fine_LA_full
            else:
                self.organizer.latent_annotations[0] = super(ConstituentSMExperiment, self).create_initial_la()
            self.save_current_la()
            self.organizer.training_reducts = None

            print("Recomputing reducts", file=self.logger)
            self.update_reducts(self.compute_reducts(self.resources[TRAINING]))
            self.stage_dict["stage"] = (3, 3, 2)
            # self.initialize_training_environment()
            # self.organizer.last_sm_cycle = 0
            # self.organizer.latent_annotations[0] = super(ConstituentSMExperiment, self).create_initial_la()

            # raise Exception("No text")

    def print_funky_listing(self, merge_sources):
        lookup = {}

        for nont_idx in range(0, self.organizer.nonterminal_map.get_counter()):
            nont = self.organizer.nonterminal_map.index_object(nont_idx)
            term = None
            if any([rule.rhs() == [] for rule in self.base_grammar.lhs_nont_to_rules(nont)]):
                print(nont, file=self.logger)
                for rule in self.base_grammar.lhs_nont_to_rules(nont):
                    print(rule, file=self.logger)
                    assert len(rule.lhs().args()) == 1 and len(rule.lhs().args()[0]) == 1
                    # rule_term = rule.lhs().args()[0][0]
                    # assert rule_term is not None
                    # if term is None:
                    #     term = rule_term
                    # else:
                    #     if term != rule_term:
                    #         print(term, rule_term)
                    #     assert term == rule_term
                    if rule.rhs() != []:
                        raise Exception("this is bad!")
                lookup[nont] = {}
                # print(merge_sources[nont_idx])
                # print(self.split_id[nont])
                for group, sources in enumerate(merge_sources[nont_idx]):
                    print("group", group, file=self.logger)
                    for source in sources:
                        for key in self.split_id[nont]:
                            if self.split_id[nont][key] - 1 == source:
                                print("\t", key, file=self.logger)
                                lookup[nont][frozenset(key[0])] = group
                                continue
                # print("lookup")
                # for key in lookup[nont]:
                #     print(key, lookup[nont][key])
        return lookup

    def patch_terminal_labeling(self, lookup):
        this_class = self

        class PatchedTerminalLabeling(TerminalLabeling):
            def __init__(self, other, lookup):
                self.other = other
                self.lookup = lookup

            def token_label(self, token):
                other_label = self.other.token_label(token)
                feat_list = token_to_features(token)
                features = this_class.induction_settings.feat_function([feat_list])
                feature_set = frozenset(features[0])
                group_idx = self.lookup.get(feature_set, 0)

                return other_label + "-group-" + str(group_idx)

        class PatchedTerminalLabeling2(TerminalLabeling):
            def __init__(self, other, lookup):
                self.other = other
                self.lookup = lookup

            def token_label(self, token):
                other_label = self.other.token_label(token)
                feat_list = token_to_features(token)
                features = this_class.induction_settings.feat_function([feat_list])
                feature_set = frozenset(features[0])
                if token.pos() + "/1" not in self.lookup:
                    return token.pos()
                if feature_set in self.lookup[token.pos() + "/1"]:
                    return other_label
                else:
                    return token.pos()

        self.terminal_labeling = PatchedTerminalLabeling2(self.induction_settings.terminal_labeling, lookup)


LABELING_STRATEGIES_BASE = ['strict', 'child'] \
                           + ['strict-markov-v-%i-h-%i' % p for p in itertools.product(range(0, 2), range(0, 4))]
LABELING_STRATEGIES = LABELING_STRATEGIES_BASE + [ '%s-spans' % s for s in LABELING_STRATEGIES_BASE]
BACKOFF = ['yes', 'auto', 'no']  # auto: use backoff for form+unk terminal


@plac.annotations(
    split=('the corpus/split to run the experiment on', 'positional', None, str, SPLITS),
    test_mode=('evaluate on test set instead of dev. set', 'flag'),
    quick=('run a small experiment (for testing/debugging)', 'flag'),
    terminal_labeling=('style of terminals in grammar', 'option', None, str, TERMINAL_LABELINGS),
    unk_threshold=('threshold for unking rare words', 'option', None, int),
    terminal_backoff=('add "all backoff" version of sentences to training/validation corpus', 'option', None, str, BACKOFF),
    recursive_partitioning=('recursive partitioning strategy', 'option', None, str),
    nonterminal_naming_scheme=('scheme for naming nonterminals', 'option', None, str, LABELING_STRATEGIES),
    no_edge_labels=('do not include edge labels in sDCP', 'flag'),
    seed=('random seed for tie-breaking after splitting', 'option', None, int),
    threads=('number of threads during expectation step (requires compilation with OpenMP flag set)', 'option', None, int),
    em_epochs=('epochs of EM before split/merge training', 'option', None, int),
    em_epochs_sm=('epochs of EM during split/merge training', 'option', None, int),
    sm_cycles=('number of split/merge cycles', 'option', None, int),
    merge_percentage=('percentage of splits that is merged', 'option', None, float),
    predicted_pos=('use predicted POS-tags for evaluation', 'flag'),
    parsing_mode=('parsing mode for evaluation', 'option', None, str,
                  [MULTI_OBJECTIVES, BASE_GRAMMAR, MAX_RULE_PRODUCT_ONLY, MULTI_OBJECTIVES_INDEPENDENT, NO_PARSING]),
    parsing_limit=('only evaluate on sentences of length up to 40', 'flag'),
    k_best=('k in k-best reranking parsing mode', 'option', None, int),
    directory=('directory in which experiment is run (default: mktemp)', 'option', None, str),
    counts_prior=('number that is added to each rule\'s expected frequency during EM training', 'option', None, float)
    )
def main(split,
         test_mode=False,
         quick=False,
         terminal_labeling='form+pos',
         unk_threshold=4,
         terminal_backoff='auto',
         recursive_partitioning="fanout-2-left-to-right",
         nonterminal_naming_scheme="child",
         no_edge_labels=False,
         seed=0,
         threads=8,
         em_epochs=20,
         em_epochs_sm=20,
         sm_cycles=4,
         merge_percentage=50.0,
         predicted_pos=False,
         parsing_mode=MULTI_OBJECTIVES,
         parsing_limit=False,
         k_best=500,
         directory=None,
         counts_prior=0.0
         ):
    """
    Run an end-to-end experiment with LCFRS/sDCP hybrid grammars on constituent parsing.
    """
    induction_settings = InductionSettings()
    induction_settings.recursive_partitioning \
        = the_recursive_partitioning_factory().get_partitioning(recursive_partitioning)[0]
    induction_settings.normalize = True
    induction_settings.disconnect_punctuation = False
    induction_settings.naming_scheme = nonterminal_naming_scheme
    induction_settings.edge_labels = not no_edge_labels
    induction_settings.isolate_pos = True

    experiment = ConstituentSMExperiment(induction_settings, directory=directory)
    experiment.organizer.seed = seed
    experiment.organizer.em_epochs = em_epochs
    experiment.organizer.em_epochs_sm = em_epochs_sm
    experiment.organizer.validator_type = "SIMPLE"
    experiment.organizer.max_sm_cycles = sm_cycles
    experiment.counts_prior = counts_prior

    experiment.organizer.disable_split_merge = False
    experiment.organizer.disable_em = False
    experiment.organizer.merge_percentage = merge_percentage
    experiment.organizer.merge_type = "PERCENT"
    experiment.organizer.threads = threads

    train, dev, test, test_input = setup_corpus_resources(split,
                                                          not test_mode,
                                                          quick,
                                                          test_pred=predicted_pos,
                                                          test_second_half=TEST_SECOND_HALF)
    experiment.resources[TRAINING] = train
    experiment.resources[VALIDATION] = dev
    experiment.resources[TESTING] = test
    experiment.resources[TESTING_INPUT] = test_input

    if "km2003" in split:
        experiment.eval_postprocess_options = ("--reversetransforms=km2003wsj",)

    if parsing_limit:
        experiment.max_sentence_length_for_parsing = 40

    experiment.k_best = k_best

    experiment.disco_dop_params["pruning_k"] = 50000
    experiment.read_stage_file()

    if terminal_backoff == 'yes':
        experiment.backoff = True
    elif terminal_backoff == 'no':
        experiment.backoff = False
    elif terminal_backoff == 'auto':
        if terminal_labeling == 'form+pos':
            experiment.backoff = True
        else:
            experiment.backoff = False

    # only effective if no terminal labeling was read from stage file
    if experiment.terminal_labeling is None:
        experiment.set_terminal_labeling(
            construct_terminal_labeling(
                terminal_labeling,
                experiment.read_corpus(experiment.resources[TRAINING]),
                threshold=unk_threshold))

    if parsing_mode == NO_PARSING:
        experiment.parsing_mode = NO_PARSING
        experiment.run_experiment()
    elif parsing_mode == MULTI_OBJECTIVES:
        experiment.parsing_mode = "discodop-multi-method"
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger,
                                                       secondary_scores=3)
        experiment.run_experiment()
    elif parsing_mode == BASE_GRAMMAR:
        experiment.k_best = 1
        experiment.organizer.project_weights_before_parsing = False
        experiment.parsing_mode = "k-best-rerank-disco-dop"
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.run_experiment()
    elif parsing_mode == MAX_RULE_PRODUCT_ONLY:
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "max-rule-prod-disco-dop"
        experiment.run_experiment()
    elif parsing_mode == MULTI_OBJECTIVES_INDEPENDENT:
        experiment.parsing_mode = "latent-viterbi-disco-dop"
        experiment.run_experiment()

        experiment.parsing_mode = "k-best-rerank-disco-dop"
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.run_experiment()

        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "variational-disco-dop"
        experiment.run_experiment()

        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "max-rule-prod-disco-dop"
        experiment.run_experiment()
    else:
        raise ValueError("Invalid parsing mod: ", parsing_mode)


if __name__ == '__main__':
    plac.call(main)


__all__ = ["ConstituentExperiment", 'MULTI_OBJECTIVES', 'BASE_GRAMMAR',
           'MAX_RULE_PRODUCT_ONLY', 'MULTI_OBJECTIVES_INDEPENDENT', 'NEGRA']
