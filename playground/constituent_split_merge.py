from __future__ import print_function
from ast import literal_eval
import copy
from collections import defaultdict
try:
    from functools32 import lru_cache
except ImportError:
    from functools import lru_cache
import os
import pickle
import sys
import subprocess
import tempfile
import itertools
from hybridtree.general_hybrid_tree import HybridTree
from parser.discodop_parser.parser import DiscodopKbestParser
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree
from parser.trace_manager.sm_trainer import build_PyLatentAnnotation
from parser.lcfrs_la import construct_fine_grammar
import plac
from grammar.induction.terminal_labeling import  PosTerminals, TerminalLabeling, FeatureTerminals, \
    FrequencyBiasedTerminalLabeling, CompositionalTerminalLabeling, FormTerminals
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from constituent.induction import fringe_extract_lcfrs, token_to_features
from constituent.construct_morph_annotation import build_nont_splits_dict, pos_cat_feats, pos_cat_and_lex_in_unary, \
    extract_feat
from constituent.discodop_adapter import TreeComparator as DiscoDopScorer
from constituent.dummy_tree import dummy_constituent_tree, flat_dummy_constituent_tree
from constituent.parse_accuracy import ParseAccuracyPenalizeFailures
import corpora.tiger_parse as tp
import corpora.negra_parse as np
import corpora.tagged_parse as tagged_parse
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.monadic_tokens import construct_constituent_token, ConstituentCategory
from playground.experiment_helpers import ScoringExperiment, CorpusFile, ScorerResource, RESULT, TRAINING, TESTING, \
    VALIDATION, TESTING_INPUT, SplitMergeExperiment
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf8')
# import codecs
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)


def setup_corpus_resources(split, dev_mode=True, quick=False, test_pred=False, test_second_half=False):
    """
    :param split: A string specifying a particular corpus and split from the literature.
    :type split: str
    :param dev_mode: If true, then the development set is used for testing.
    :type dev_mode: bool
    :param quick: If true, then a smaller version of the corpora are returned.
    :type quick: bool
    :param test_pred: If true, then predicted POS tags are used for testing.
    :type test_pred: bool
    :return: A tuple with train/dev/test (in this order) of type CorpusResource
    """
    if split == "SPMRL":
        # all files are from SPMRL shared task

        corpus_type = corpus_type_test = "TIGERXML"
        train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train/train.German.gold.xml'
        train_start = 1
        train_filter = None
        train_limit = 40474
        train_exclude = validation_exclude = test_exclude = test_input_exclude = [7561, 17632, 46234, 50224]

        validation_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
        validation_start = 40475
        validation_size = validation_start + 4999
        validation_filter = None

        if dev_mode:
            test_start = test_input_start = validation_start
            test_limit = test_input_limit = validation_size
            test_path = test_input_path \
                = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
        else:
            test_start = test_input_start = 45475
            test_limit = test_input_limit = test_start + 4999
            test_path = test_input_path \
                = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/test/test.German.gold.xml'
        test_filter = test_input_filter = None

        if quick:
            train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train5k/train5k.German.gold.xml'
            train_limit = train_start + 2000
            validation_size = validation_start + 200
            test_limit = test_input_limit = test_start + 200
    #
    elif split == "HN08":
        # files are based on the scripts in Coavoux's mind the gap 1.0
        # where we commented out `rm -r tiger21 tiger22 marmot_tags` in generate_tiger_data.sh

        corpus_type = corpus_type_test = "EXPORT"
        base_path = "../res/TIGER/tiger21"
        train_start = 1
        train_limit = 50474

        train_path = os.path.join(base_path, "tigertraindev_root_attach.export")

        def train_filter(x):
            return x % 10 >= 2

        train_exclude = [7561, 17632, 46234, 50224]

        validation_start = 1
        validation_size = 50471
        validation_exclude = train_exclude
        validation_path = os.path.join(base_path, "tigerdev_root_attach.export")
        validation_exclude = train_exclude

        def validation_filter(sent_id):
            return sent_id % 10 == 1

        if not dev_mode:
            test_start = test_input_start = 1  # validation_size  # 40475
            if test_second_half:
                test_start = test_input_start = 25240
            test_limit = test_input_limit = 50474
            # test_limit = 200 * 5 // 4
            test_exclude = test_input_exclude = train_exclude
            test_path = os.path.join(base_path, "tigertest_root_attach.export")

            def test_filter(sent_id):
                return sent_id % 10 == 0

            if test_pred:
                corpus_type_test = "WORD/POS"
                test_input_start = 0
                if test_second_half:
                    test_input_start = 2524 - 1
                # predicted by MATE trained on tigerHN08 train + dev
                test_input_path = '../res/TIGER/tigerHN08-test.train+dev.pred_tags.raw'
                test_input_filter = None
            else:
                test_input_path = test_path
                test_input_filter = test_filter

        else:
            test_start = test_input_start = 1
            if test_second_half:
                test_start = test_input_start = 25241
            test_limit = test_input_limit = 50474
            test_exclude = test_input_exclude = train_exclude
            test_path = validation_path
            test_filter = validation_filter

            if test_pred:
                corpus_type_test = "WORD/POS"
                test_input_start = 0
                if test_second_half:
                    test_input_start = 2524
                # predicted by MATE trained on tigerHN08 train
                test_input_path = '../res/TIGER/tigerHN08-dev.train.pred_tags.raw'
                test_input_filter = None
            else:
                test_input_path = validation_path
                test_input_filter = test_filter

        if quick:
            train_limit = 5000 * 5 // 4
            validation_size = 200 * 10 // 1
            TEST_LIMIT = 200
            test_limit = test_input_limit = TEST_LIMIT * 10 // 1
            if test_pred:
                test_input_limit = TEST_LIMIT + 1
    #
    elif "WSJ" in split:
        # based on Kilian Evang's dptb.tar.bz2

        corpus_type = corpus_type_test = "EXPORT"
        corpus_path_original = "../res/WSJ/ptb-discontinuous/dptb7.export"
        corpus_path_km2003 = "../res/WSJ/ptb-discontinuous/dptb7-km2003wsj.export"

        # obtain the km2003 version from by running
        # discodop treetransforms --transforms=km2003wsj corpus_path_original corpus_path_km2003

        if "km2003" in split:
            corpus_path = corpus_path_km2003
        else:
            corpus_path = corpus_path_original

        train_path = validation_path = test_path = test_input_path = corpus_path
        train_exclude = validation_exclude = test_exclude = test_input_exclude = []
        train_filter = validation_filter = test_filter = test_input_filter = None

        # sections 2-21
        train_start = 3915
        train_limit = 43746

        # section 24
        validation_start = 47863
        validation_size = 49208

        if not dev_mode:
            # section 23
            test_start = test_input_start = 45447
            test_limit = test_input_limit = 47862
        else:
            test_start = test_input_start = validation_start
            test_limit = test_input_limit = validation_size

        if quick:
            train_limit = train_start + 2000
            validation_size = validation_start + 200
            test_limit = test_input_limit = test_start + 200
    else:
        raise ValueError("Unknown split: " + str(split))

    train = CorpusFile(path=train_path, start=train_start, end=train_limit, exclude=train_exclude, filter=train_filter,
                       type=corpus_type)
    dev = CorpusFile(path=validation_path, start=validation_start, end=validation_size, exclude=validation_exclude,
                     filter=validation_filter, type=corpus_type)
    test = CorpusFile(path=test_path, start=test_start, end=test_limit, exclude=test_exclude, filter=test_filter,
                      type=corpus_type)
    test_input = CorpusFile(path=test_input_path,
                            start=test_input_start,
                            end=test_input_limit,
                            exclude=test_input_exclude,
                            filter=test_input_filter,
                            type=corpus_type_test)

    return train, dev, test, test_input


# if not os.path.isfile(terminal_labeling_path):
#     terminal_labeling = FormPosTerminalsUnk(get_train_corpus(), 10)
#     pickle.dump(terminal_labeling, open(terminal_labeling_path, "wb"))
# else:
#     terminal_labeling = pickle.load(open(terminal_labeling_path, "rb"))

# terminal_labeling = PosTerminals()

# terminal_labeling = FeatureTerminals(token_to_features,
#                                      feature_filter=lambda x: pos_cat_and_lex_in_unary(x, no_function=True))

# fine_terminal_labeling = FeatureTerminals(token_to_features,
#                                           feature_filter=lambda x: pos_cat_and_lex_in_unary(x, no_function=True))


def my_feature_filter(elem):
    base_feats = ["number", "person", "tense", "mood", "case", "degree", "category", "pos", "gender"]
    feat_set = {feat: value for feat, value in elem[0]}
    if "pos" in feat_set and feat_set["pos"] in {"APPR", "APPRART"}:
        return extract_feat(elem[0], features=base_feats + ["lemma"])
    return extract_feat(elem[0], features=base_feats)


FINE_TERMINAL_LABELING = FeatureTerminals(token_to_features, feature_filter=my_feature_filter)
FINE_TERMINAL_LABELING = CompositionalTerminalLabeling(FormTerminals(), PosTerminals())
FALLBACK_TERMINAL_LABELING = PosTerminals()

DEFAULT_RARE_WORD_THRESHOLD = 10


def terminal_labeling(corpus, threshold=DEFAULT_RARE_WORD_THRESHOLD):
    return FrequencyBiasedTerminalLabeling(FINE_TERMINAL_LABELING, FALLBACK_TERMINAL_LABELING, corpus, threshold)


# SPLIT = "SPMRL"
SPLIT = "HN08"
# SPLIT = "WSJ"
# SPLIT = "WSJ-km2003"

DEV_MODE = True
QUICK = False

MULTI_OBJECTIVES = True
BASE_GRAMMAR = False  # use base grammar for parsing (no annotations LA)
MAX_RULE_PRODUCT_ONLY = False
LENGTH_40 = True
TEST_SECOND_HALF = True

FANOUT = 2
RECURSIVE_PARTITIONING \
    = the_recursive_partitioning_factory().getPartitioning('fanout-' + str(FANOUT) + '-left-to-right')[0]

MAX_SENTENCE_LENGTH = 5000
EM_EPOCHS = 20
EM_EPOCHS_SM = 20
SEED = 0
MERGE_PERCENTAGE = 50.0
SM_CYCLES = 4
THREADS = 1  # 0
K_BEST = 500

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
        self.disconnect_punctuation = True
        self.normalize = False
        self.feature_la = False
        self.feat_function = FeatureFunction()

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
        self.terminal_labeling = induction_settings.terminal_labeling
        self.eval_postprocess_options = None

        self.discodop_scorer = DiscoDopScorer()
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
        dcp = The_DCP_evaluator(derivation).getEvaluation()
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

        prm = "../util/proper.prm"

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
            return np.hybridtrees_to_sentence_names([obj], number, MAX_SENTENCE_LENGTH)
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
        ScoringExperiment.read_stage_file(self)

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

    def induce_from(self, obj):
        if not obj.complete() or obj.empty_fringe():
            return None, None
        part = self.induction_settings.recursive_partitioning(obj)

        features = defaultdict(lambda: 0) if self.induction_settings.feature_la else None

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


@plac.annotations(
    directory=('directory in which experiment is run', 'option', None, str)
    )
def main(directory=None):
    induction_settings = InductionSettings()
    induction_settings.recursive_partitioning = RECURSIVE_PARTITIONING
    induction_settings.normalize = True
    induction_settings.disconnect_punctuation = False
    induction_settings.naming_scheme = 'strict-markov-v-1-h-1'
    induction_settings.isolate_pos = True

    experiment = ConstituentSMExperiment(induction_settings, directory=directory)
    experiment.organizer.seed = SEED
    experiment.organizer.em_epochs = EM_EPOCHS
    experiment.organizer.em_epochs_sm = EM_EPOCHS_SM
    experiment.organizer.validator_type = "SIMPLE"
    experiment.organizer.max_sm_cycles = SM_CYCLES

    experiment.organizer.disable_split_merge = False
    experiment.organizer.disable_em = False
    experiment.organizer.merge_percentage = MERGE_PERCENTAGE
    experiment.organizer.merge_type = "PERCENT"
    experiment.organizer.threads = 8

    train, dev, test, test_input = setup_corpus_resources(SPLIT, DEV_MODE, QUICK, test_second_half=TEST_SECOND_HALF)
    experiment.resources[TRAINING] = train
    experiment.resources[VALIDATION] = dev
    experiment.resources[TESTING] = test
    experiment.resources[TESTING_INPUT] = test_input

    if "km2003" in SPLIT:
        experiment.eval_postprocess_options = ("--reversetransforms=km2003wsj",)

    if LENGTH_40:
        experiment.max_sentence_length_for_parsing = 40

    experiment.k_best = K_BEST
    experiment.backoff = True

    backoff_threshold = 4
    induction_settings.terminal_labeling = terminal_labeling(experiment.read_corpus(experiment.resources[TRAINING]),
                                                             threshold=backoff_threshold)

    experiment.terminal_labeling = induction_settings.terminal_labeling
    experiment.disco_dop_params["pruning_k"] = 50000
    experiment.read_stage_file()

    if MULTI_OBJECTIVES:
        experiment.parsing_mode = "discodop-multi-method"
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger,
                                                       secondary_scores=3)
        experiment.run_experiment()
    elif BASE_GRAMMAR:
        experiment.k_best = 1
        experiment.organizer.project_weights_before_parsing = False
        experiment.parsing_mode = "k-best-rerank-disco-dop"
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.run_experiment()
    elif MAX_RULE_PRODUCT_ONLY:
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "max-rule-prod-disco-dop"
        experiment.run_experiment()
    else:
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


if __name__ == '__main__':
    plac.call(main)
