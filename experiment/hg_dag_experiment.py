import copy
import itertools
import json
import os
import subprocess
import tempfile

import plac

import corpora.negra_parse as np
import corpora.tagged_parse as tagged_parse
from constituent.dag_induction import direct_extract_lcfrs_from_prebinarized_corpus, BasicNonterminalLabeling
from constituent.dummy_tree import flat_dummy_constituent_tree
from experiment.base_experiment import ScoringExperiment
from experiment.hg_constituent_experiment import MULTI_OBJECTIVES, BASE_GRAMMAR, MAX_RULE_PRODUCT_ONLY, \
    MULTI_OBJECTIVES_INDEPENDENT, NEGRA, ScorerAndWriter
from experiment.resources import CorpusFile, RESULT, TESTING, TESTING_INPUT, VALIDATION, TRAINING
from experiment.split_merge_experiment import SplitMergeExperiment
from grammar.induction.terminal_labeling import PosTerminals, FrequencyBiasedTerminalLabeling, \
    CompositionalTerminalLabeling, FormTerminals, deserialize_labeling
from grammar.linearization import linearize
from graphs.schick_parser_rtg_import import read_rtg
from hybridtree.general_hybrid_tree import HybridDag
from hybridtree.monadic_tokens import construct_constituent_token
from parser.discodop_parser.parser import DiscodopKbestParser
from parser.sDCPevaluation.evaluator import dcp_to_hybriddag, DCP_evaluator
from parser.supervised_trainer.trainer import PyDerivationManager

TRAINING_BIN = 'TRAINING_BIN'
TEST_SECOND_HALF = False

FINE_TERMINAL_LABELING = CompositionalTerminalLabeling(FormTerminals(), PosTerminals())
FALLBACK_TERMINAL_LABELING = PosTerminals()
DEFAULT_RARE_WORD_THRESHOLD = 10

DAG_CORPUS = 'res/tiger/tiger_full_with_sec_edges.export'
DAG_CORPUS_BIN = 'res/tiger/tiger_full_with_sec_edges_bin_h1_v1.export'

SCHICK_PARSER_JAR = 'HypergraphReduct-1.0-SNAPSHOT.jar'


def terminal_labeling(corpus, threshold=DEFAULT_RARE_WORD_THRESHOLD):
    return FrequencyBiasedTerminalLabeling(FINE_TERMINAL_LABELING, FALLBACK_TERMINAL_LABELING, corpus, threshold)


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
    if not os.path.exists(DAG_CORPUS):
        print('run the following command to create an export corpus with dags:')
        print('\tPYTHONPATH=. util/tiger_dags_to_negra.py ' +
              'res/tiger/tiger_release_aug07.corrected.16012013.xml '
              + DAG_CORPUS + ' 1 50474')

    if not os.path.exists(DAG_CORPUS_BIN):
        print('run the following command to binarize the export corpus with dags:')
        print("discodop treetransforms --binarize -v 1 -h 1 " + DAG_CORPUS + " " + DAG_CORPUS_BIN)
        # _, DAG_CORPUS_BIN = tempfile.mkstemp(prefix='corpus_bin_', suffix='.export')
        # subprocess.call(["discodop", "treetransforms", "--binarize", "-v", "1", "-h", "1", DAG_CORPUS, DAG_CORPUS_BIN])

    if split == "SPMRL":
        # all files are from SPMRL shared task

        corpus_type = corpus_type_test = "TIGERXML"
        train_path = 'res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train/train.German.gold.xml'
        train_start = 1
        train_filter = None
        train_limit = 40474
        train_exclude = validation_exclude = test_exclude = test_input_exclude = [7561, 17632, 46234, 50224]

        validation_path = 'res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
        validation_start = 40475
        validation_size = validation_start + 4999
        validation_filter = None

        if dev_mode:
            test_start = test_input_start = validation_start
            test_limit = test_input_limit = validation_size
            test_path = test_input_path \
                = 'res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
        else:
            test_start = test_input_start = 45475
            test_limit = test_input_limit = test_start + 4999
            test_path = test_input_path \
                = 'res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/test/test.German.gold.xml'
        test_filter = test_input_filter = None

        if quick:
            train_path = 'res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train5k/train5k.German.gold.xml'
            train_limit = train_start + 2000
            validation_size = validation_start + 200
            test_limit = test_input_limit = test_start + 200
    #
    elif split == "HN08":
        # files are based on the scripts in Coavoux's mind the gap 1.0
        # where we commented out `rm -r tiger21 tiger22 marmot_tags` in generate_tiger_data.sh

        corpus_type = corpus_type_test = "EXPORT"
        base_path = "res/TIGER/tiger21"
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
                test_input_path = 'res/TIGER/tigerHN08-test.train+dev.pred_tags.raw'
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
                test_input_path = 'res/TIGER/tigerHN08-dev.train.pred_tags.raw'
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
    else:
        raise ValueError("Unsupported split: " + split)

    corpus_type = 'EXPORT'

    if not test_pred:
        corpus_type_test = 'EXPORT'

    train = CorpusFile(path=DAG_CORPUS, start=train_start, end=train_limit, exclude=train_exclude,
                       filter=train_filter,
                       type=corpus_type)
    train_bin = CorpusFile(path=DAG_CORPUS_BIN, start=train_start, end=train_limit, exclude=train_exclude,
                           filter=train_filter,
                           type=corpus_type)
    dev = CorpusFile(path=DAG_CORPUS, start=validation_start, end=validation_size, exclude=validation_exclude,
                     filter=validation_filter, type=corpus_type)
    test = CorpusFile(path=DAG_CORPUS, start=test_start, end=test_limit, exclude=test_exclude, filter=test_filter,
                      type=corpus_type)
    test_input = CorpusFile(path=DAG_CORPUS,
                            start=test_input_start,
                            end=test_input_limit,
                            exclude=test_input_exclude,
                            filter=test_input_filter,
                            type=corpus_type_test)

    return train, train_bin, dev, test, test_input


class DagExperiment(ScoringExperiment, SplitMergeExperiment):
    def __init__(self, induction_settings, directory=None):
        ScoringExperiment.__init__(self, directory)
        SplitMergeExperiment.__init__(self)
        self.induction_settings = induction_settings
        self.backoff = False
        self.terminal_labeling = induction_settings.terminal_labeling
        self.parser = None
        # self.use_output_counter = False
        self.serialization_type = NEGRA
        self.strip_vroot = False
        self.backoff_factor = 1.0
        self.delimiter = ' : '

    def obtain_sentence(self, obj):
        if isinstance(obj, HybridDag):
            sentence = obj.full_yield(), obj.id_yield(), \
                       obj.full_token_yield(), obj.token_yield()
            return sentence
        elif isinstance(obj, list):
            return [i for i in range(len(obj))], [i for i in range(len(obj))], obj, obj
        else:
            raise ValueError("Unsupported obj type", type(obj), "instance", obj)

    def obtain_label(self, hybrid_dag):
        return hybrid_dag.sent_label()

    def compute_fallback(self, sentence, label=None):
        full_yield, id_yield, full_token_yield, token_yield = sentence
        return flat_dummy_constituent_tree(token_yield, full_token_yield, 'NP', 'S', label)

    def read_corpus(self, resource):
        if resource.type == "EXPORT":
            return self.read_corpus_export(resource)
        elif resource.type == "WORD/POS":
            return self.read_corpus_tagged(resource)
        else:
            raise ValueError("Unsupported resource type " + resource.type)

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
            enc=encoding,
            disconnect_punctuation=self.induction_settings.disconnect_punctuation,
            add_vroot=False,  # todo True ?!
            mode=mode,
            secedge=True)

    def read_corpus_tagged(self, resource):
        return itertools.islice(tagged_parse.parse_tagged_sentences(resource.path), resource.start, resource.limit)

    def additional_induction_params(self, obj):
        return {'hybrid_dag_bin': self.resources_data['binarized_corpus_dict'].get(obj.sent_label())}

    def induction_preparation(self):
        self.resources_data['binarized_corpus_dict'] = {dag.sent_label(): dag for dag in
                                                        self.read_corpus(self.resources[TRAINING_BIN])}

    def induce_from(self, obj, **kwargs):
        hybrid_dag_bin = kwargs['hybrid_dag_bin']

        dag_grammar = direct_extract_lcfrs_from_prebinarized_corpus(hybrid_dag_bin)

        if self.backoff:
            self.terminal_labeling.backoff_mode = True

            dag_grammar_backoff \
                = direct_extract_lcfrs_from_prebinarized_corpus(hybrid_dag_bin,
                                                                term_labeling=self.induction_settings.terminal_labeling,
                                                                nont_labeling=self.induction_settings.nont_labeling)
            dag_grammar.add_gram(dag_grammar_backoff)

            self.terminal_labeling.backoff_mode = False

        if False and len(obj.token_yield()) == 1:
            print(obj, map(str, obj.token_yield()), file=self.logger)
            print(dag_grammar, file=self.logger)

        return dag_grammar, None

    def initialize_parser(self):
        assert "disco-dop" in self.parsing_mode
        self.parser = DiscodopKbestParser(grammar=self.base_grammar,
                                          k=self.k_best,
                                          beam_beta=self.disco_dop_params["beam_beta"],
                                          beam_delta=self.disco_dop_params["beam_delta"],
                                          pruning_k=self.disco_dop_params["pruning_k"],
                                          cfg_ctf=self.disco_dop_params["cfg_ctf"])

    def read_stage_file(self):
        ScoringExperiment.read_stage_file(self)
        if "terminal_labeling" in self.stage_dict:
            terminal_labeling_path = self.stage_dict["terminal_labeling"]
            with open(terminal_labeling_path, "r") as tlf:
                self.terminal_labeling = deserialize_labeling(json.load(tlf))
                self.induction_settings.terminal_labeling = self.terminal_labeling

        if "training_reducts" in self.stage_dict:
            self.organizer.training_reducts = PyDerivationManager(self.base_grammar) # self.terminal_labeling)
            self.organizer.training_reducts.load_traces_from_file(
                bytes(self.stage_dict["training_reducts"], encoding="utf-8"))

        if "validation_reducts" in self.stage_dict:
            self.organizer.validation_reducts = PyDerivationManager(self.base_grammar)  # self.terminal_labeling)
            self.organizer.validation_reducts.load_traces_from_file(
                bytes(self.stage_dict["validation_reducts"], encoding="utf-8"))

        SplitMergeExperiment.read_stage_file(self)

    def evaluate(self, result_resource, _):
        print('normalize results with treetools and discodop', file=self.logger)

        if self.induction_settings.normalize:
            ref_rn = self.normalize_corpus(result_resource.reference.path, disco_options=self.eval_postprocess_options)
            sys_rn = self.normalize_corpus(result_resource.path, disco_options=self.eval_postprocess_options)
            sys_secs = [self.normalize_corpus(sec.path, disco_options=self.eval_postprocess_options)
                        for sec in result_resource.secondaries]
        else:
            ref_rn = result_resource.reference.path
            sys_rn = result_resource.path
            sys_secs = [sec.path for sec in result_resource.secondaries]

        prm = "util/proper.prm"

        def run_eval(sys_path, mode):
            print(mode)
            print('running discodop evaluation on gold:', ref_rn, ' and sys:', sys_path,
                  "with", os.path.split(prm)[1], file=self.logger)
            output = subprocess.Popen(["discodop", "eval", ref_rn, sys_path, prm, '--secedges=all'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT).communicate()
            print(str(output[0], encoding='utf-8'), file=self.logger)

        run_eval(sys_rn, "DEFAULT")

        for i, sec in enumerate(sys_secs):
            run_eval(sec, self.parser.secondaries[i])

    def serialize(self, obj):
        if self.serialization_type == NEGRA:
            # if self.use_output_counter:
            #     self.output_counter += 1
            #     number = self.output_counter
            # else:
            label = self.obtain_label(obj)
            if label.startswith('s'):
                number = int(label[1:])
            else:
                number = int(label)
            return np.serialize_hybridtrees_to_negra([obj], number, 500, use_sentence_names=False)
        raise ValueError("Unsupported serialization type", self.serialization_type)

    def print_config(self, file=None):
        if file is None:
            file = self.logger
        ScoringExperiment.print_config(self, file=file)
        print(self.induction_settings, file=file)
        print("k-best", self.k_best, file=file)
        print("Serialization type", self.serialization_type, file=file)
        # print("Output counter", self.use_output_counter, "start", self.output_counter, file=file)
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
        _, terminal_labeling_path = tempfile.mkstemp(dir=self.directory, suffix=".lexicon")
        with open(terminal_labeling_path, 'w') as fd:
            self.stage_dict["terminal_labeling"] = terminal_labeling_path
            json.dump(self.terminal_labeling.serialize(), fd)
        self.write_stage_file()

    def __corpus_linearization_and_reducts(self, corpus, corpus_path, grammar_path, reduct_dir, terminal_enc, frequency,
                                           manager):
        # linearization of corpus
        corpus_json = np.export_corpus_to_json(corpus, terminal_enc, self.terminal_labeling, delimiter=self.delimiter)

        with open(corpus_path, 'w') as data_file:
            json.dump(corpus_json, data_file)

        # computation of reducts with schick parser
        p = subprocess.Popen([' '.join(
            ["java", "-jar", os.path.join("util", SCHICK_PARSER_JAR), 'reduct', '-g', grammar_path, '-t', corpus_path,
             "--input-format", "json", "-o", reduct_dir])], shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

        while True:
            next_line = p.stdout.readline()
            if next_line == b'' and p.poll() is not None:
                break
            print(next_line.decode('unicode_escape'), end='')

        p.wait()
        p.stdout.close()
        assert p.returncode == 0

        # import reducts to derivation (trace) manager
        rtgs = []

        def decode_nonterminals(s):
            return manager.get_nonterminal_map().index_object(int(s))

        for i in range(1, len(corpus) + 1):
            rtgs.append(read_rtg(os.path.join(reduct_dir, str(i) + '.gra'),
                                 symbol_offset=-1,
                                 rule_prefix='r',
                                 process_nonterminal=decode_nonterminals))

        # derivation_manager.get_nonterminal_map().print_index()
        manager.convert_rtgs_to_hypergraphs(rtgs, frequency)

    def compute_reducts(self, resource):
        corpus = self.read_corpus(resource)
        if self.strip_vroot:
            for tree in corpus:
                tree.strip_vroot()

        # setting up files
        _, grammar_path = tempfile.mkstemp(suffix='.json', dir=self.directory, prefix='base_grammar')
        _, corpus_path = tempfile.mkstemp(suffix='.json', dir=self.directory, prefix='corpus')
        reduct_dir = tempfile.mkdtemp(dir=self.directory, prefix='reducts')

        derivation_manager = PyDerivationManager(self.base_grammar)

        # linearization of grammar
        with open(grammar_path, 'w') as grammar_file:
            nonterminal_enc, terminal_enc = linearize(self.base_grammar, self.induction_settings.nont_labeling,
                                                      self.terminal_labeling, grammar_file,
                                                      delimiter=self.delimiter,
                                                      nonterminal_encoder=derivation_manager.get_nonterminal_map())

        # derivation_manager.serialize(bytes('/tmp/reduct_manager.trace', encoding='utf8'))

        frequency = self.backoff_factor if self.backoff else 1.0
        self.__corpus_linearization_and_reducts(corpus, corpus_path, grammar_path, reduct_dir, terminal_enc, frequency,
                                                derivation_manager)

        if self.backoff:
            # setting up files for backoff corpus
            _, corpus_path_backoff = tempfile.mkstemp(suffix='.json', dir=self.directory, prefix='corpus_backoff')
            reducts_backoff = tempfile.mkdtemp(dir=self.directory, prefix='reducts_backoff')

            self.terminal_labeling.backoff_mode = True
            self.__corpus_linearization_and_reducts(corpus, corpus_path_backoff, grammar_path, reducts_backoff,
                                                    terminal_enc, 1.0, derivation_manager)
            self.terminal_labeling.backoff_mode = False

        return derivation_manager

    def parsing_preprocess(self, obj):
        if isinstance(obj, HybridDag):
            if self.strip_vroot:
                assert False and "vroot stripping is not supported"
                obj.strip_vroot()
            parser_input = self.terminal_labeling.prepare_parser_input(obj.token_yield())
            # print(parser_input)
            return parser_input
        else:
            return self.terminal_labeling.prepare_parser_input(obj)

    def parsing_postprocess(self, sentence, derivation, label=None):
        full_yield, id_yield, full_token_yield, token_yield = sentence

        dag = HybridDag(label)
        punctuation_positions = [i + 1 for i, idx in enumerate(full_yield)
                                 if idx not in id_yield]

        cleaned_tokens = copy.deepcopy(full_token_yield)
        dcp_term = DCP_evaluator(derivation).getEvaluation()

        try:
            dcp_to_hybriddag(dag, dcp_term, cleaned_tokens, False, construct_constituent_token,
                             punct_positions=punctuation_positions)
        except AssertionError:
            return self.compute_fallback(sentence, label)

        if self.strip_vroot:
            assert False and "vroot stripping is not supported"
            dag.strip_vroot()

        if dag.topological_order() is None:
            print('WARNING: dcp evaluation yields cyclic graph', dag.sent_label())
            return self.compute_fallback(sentence, label)
        return dag


class InductionSettings:
    def __init__(self):
        self.normalize = False
        self.disconnect_punctuation = True
        self.terminal_labeling = PosTerminals()
        # self.nont_labeling = NonterminalsWithFunctions()
        self.nont_labeling = BasicNonterminalLabeling()
        self.binarize = True
        self.isolate_pos = True
        self.hmarkov = 0
        self.use_discodop_binarization = False
        self.discodop_binarization_params = ["--headrules=util/negra.headrules",
                                             "--binarize",
                                             "-h 1",
                                             "-v 1"
                                             ]

    def __str__(self):
        __str = "Induction Settings {\n"
        for key in self.__dict__:
            if not key.startswith("__") and key not in []:
                __str += "\t" + key + ": " + str(self.__dict__[key]) + "\n"
        return __str + "}"


@plac.annotations(
    test_mode=('evaluate on test set instead of dev. set', 'flag'),
    unk_threshold=('threshold for unking rare words', 'option', None, int),
    h_markov=('horizontal Markovization', 'option', None, int),
    v_markov=('vertical Markovization', 'option', None, int),
    quick=('run a small experiment (for testing/debugging)', 'flag'),
    seed=('random seed for tie-breaking after splitting', 'option', None, int),
    threads=(
            'number of threads during expectation step (requires compilation with OpenMP flag set)', 'option', None,
            int),
    em_epochs=('epochs of EM before split/merge training', 'option', None, int),
    em_epochs_sm=('epochs of EM during split/merge training', 'option', None, int),
    sm_cycles=('number of split/merge cycles', 'option', None, int),
    merge_percentage=('percentage of splits that is merged', 'option', None, float),
    predicted_pos=('use predicted POS-tags for evaluation', 'flag'),
    parsing_mode=('parsing mode for evaluation', 'option', None, str,
                  [MULTI_OBJECTIVES, BASE_GRAMMAR, MAX_RULE_PRODUCT_ONLY, MULTI_OBJECTIVES_INDEPENDENT]),
    parsing_limit=('only evaluate on sentences of length up to 40', 'flag'),
    k_best=('k in k-best reranking parsing mode', 'option', None, int),
    directory=('directory in which experiment is run (default: mktemp)', 'option', None, str),
)
def main(split,
         test_mode=False,
         quick=False,
         unk_threshold=4,
         h_markov=1,
         v_markov=1,
         seed=0,
         threads=8,
         em_epochs=20,
         em_epochs_sm=20,
         sm_cycles=5,
         merge_percentage=50.0,
         predicted_pos=False,
         parsing_mode=MULTI_OBJECTIVES,
         parsing_limit=False,
         k_best=500,
         directory=None
         ):
    """
    Run an end-to-end experiment with an LCFRS/DCP hybrid grammar on dag constituent parsing.
    The LCFRS is induced with disco-dop; the DCP is copying to model reentrency.
    """
    induction_settings = InductionSettings()
    induction_settings.disconnect_punctuation = False
    induction_settings.normalize = False # todo normalize
    induction_settings.use_discodop_binarization = True
    binarization_settings = ["--headrules=" + ("util/negra.headrules" if split in ["SPMRL", "HN08"]
                                               else "util/ptb.headrules"),
                             "--binarize",
                             "-h " + str(h_markov),
                             "-v " + str(v_markov)]
    induction_settings.discodop_binarization_params = binarization_settings

    experiment = DagExperiment(induction_settings, directory=directory)

    train, train_bin, dev, test, test_input = setup_corpus_resources(split, not test_mode, quick, predicted_pos,
                                                                     TEST_SECOND_HALF)
    experiment.resources[TRAINING] = train
    experiment.resources[TRAINING_BIN] = train_bin
    experiment.resources[VALIDATION] = dev
    experiment.resources[TESTING] = test
    experiment.resources[TESTING_INPUT] = test_input

    if "km2003" in split:
        experiment.eval_postprocess_options = ("--reversetransforms=km2003wsj",)

    if parsing_limit:
        experiment.max_sentence_length_for_parsing = 40

    experiment.backoff = True
    experiment.organizer.validator_type = "SIMPLE"
    experiment.organizer.project_weights_before_parsing = True
    experiment.organizer.disable_em = False
    experiment.organizer.disable_split_merge = False
    experiment.organizer.seed = seed
    experiment.organizer.em_epochs = em_epochs
    experiment.organizer.merge_percentage = merge_percentage
    experiment.organizer.em_epochs_sm = em_epochs_sm
    experiment.organizer.max_sm_cycles = sm_cycles
    experiment.organizer.threads = threads
    experiment.oracle_parsing = False
    experiment.k_best = k_best
    experiment.disco_dop_params["pruning_k"] = 50000
    experiment.read_stage_file()

    # only effective if no terminal labeling was read from stage file
    if experiment.terminal_labeling is None:
        # StanfordUNKing(experiment.read_corpus(experiment.resources[TRAINING]))
        experiment.set_terminal_labeling(terminal_labeling(experiment.read_corpus(experiment.resources[TRAINING]),
                                                           threshold=unk_threshold))
    if parsing_mode == MULTI_OBJECTIVES:
        experiment.parsing_mode = "discodop-multi-method"
        experiment.resources[RESULT] = ScorerAndWriter(experiment, directory=experiment.directory,
                                                       logger=experiment.logger, secondary_scores=3)
        experiment.run_experiment()
    elif parsing_mode == BASE_GRAMMAR:
        experiment.k_best = 1
        experiment.organizer.project_weights_before_parsing = False
        experiment.parsing_mode = "k-best-rerank-disco-dop"
        experiment.resources[RESULT] = ScorerAndWriter(experiment, directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.run_experiment()
    elif parsing_mode == MAX_RULE_PRODUCT_ONLY:
        experiment.resources[RESULT] = ScorerAndWriter(experiment, directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "max-rule-prod-disco-dop"
        experiment.run_experiment()
    elif parsing_mode == MULTI_OBJECTIVES_INDEPENDENT:
        experiment.parsing_mode = "latent-viterbi-disco-dop"
        experiment.run_experiment()

        experiment.parsing_mode = "k-best-rerank-disco-dop"
        experiment.resources[RESULT] = ScorerAndWriter(experiment, directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.run_experiment()

        experiment.resources[RESULT] = ScorerAndWriter(experiment, directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "variational-disco-dop"
        experiment.run_experiment()

        experiment.resources[RESULT] = ScorerAndWriter(experiment, directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "max-rule-prod-disco-dop"
        experiment.run_experiment()
    else:
        raise ValueError("Invalid parsing mod: ", parsing_mode)


if __name__ == '__main__':
    plac.call(main)

__all__ = []
