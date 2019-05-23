from __future__ import print_function
import tempfile
import subprocess
from parser.discodop_parser.parser import DiscodopKbestParser
try:
    from parser.gf_parser.gf_interface import GFParser_k_best
except ImportError:
    print("The Grammatical Framework is not installed properly – the GFParser is unavailable.")
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
import plac
from constituent.induction import direct_extract_lcfrs, BasicNonterminalLabeling, \
    direct_extract_lcfrs_from_prebinarized_corpus
from grammar.induction.terminal_labeling import PosTerminals, FrequencyBiasedTerminalLabeling, \
    FormTerminals, CompositionalTerminalLabeling
from experiment.resources import TRAINING, VALIDATION, TESTING, TESTING_INPUT, RESULT, CorpusFile
from experiment.split_merge_experiment import SplitMergeExperiment
from experiment.hg_constituent_experiment import ConstituentExperiment, ScoringExperiment, ScorerAndWriter, \
    setup_corpus_resources, MULTI_OBJECTIVES, MULTI_OBJECTIVES_INDEPENDENT, BASE_GRAMMAR, MAX_RULE_PRODUCT_ONLY, \
    NO_PARSING

TEST_SECOND_HALF = False  # parse second half of test set

# FINE_TERMINAL_LABELING = FeatureTerminals(token_to_features, feature_filter=my_feature_filter)
# FINE_TERMINAL_LABELING = FormTerminals()
FINE_TERMINAL_LABELING = CompositionalTerminalLabeling(FormTerminals(), PosTerminals())
FALLBACK_TERMINAL_LABELING = PosTerminals()
DEFAULT_RARE_WORD_THRESHOLD = 10


def terminal_labeling(corpus, threshold=DEFAULT_RARE_WORD_THRESHOLD):
    return FrequencyBiasedTerminalLabeling(FINE_TERMINAL_LABELING, FALLBACK_TERMINAL_LABELING, corpus, threshold)


class InductionSettings:
    def __init__(self):
        self.normalize = False
        self.disconnect_punctuation = True
        self.terminal_labeling = PosTerminals()
        self.edge_labels = True
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


class LCFRSExperiment(ConstituentExperiment, SplitMergeExperiment):
    """
    Holds state and methods of a LCFRS-LA parsing experiment.
    """
    def __init__(self, induction_settings, directory=None, filters=None):
        ConstituentExperiment.__init__(self, induction_settings, directory=directory, filters=filters)
        SplitMergeExperiment.__init__(self)

        self.strip_vroot = False
        self.k_best = 500
        self.resources_data['disco_binarized_corpus'] = None
        self.dummy_flag = True

    @staticmethod
    def __valid_tree(obj):
        return obj.complete() and not obj.empty_fringe()

    def induce_from_disco_binarized(self, htree):
        """
        :type htree: HybridTree
        Induces a binarized LCFRS from a binarized version of htree, which is obtained by discodop.
        NB: the resulting grammar parses htree, due to a suitable choice of the tree component of the induced grammar.
        """
        self.run_discodop_binarization()
        htree_bin = None
        for _htree in self.resources_data['disco_binarized_corpus']:
            if _htree.sent_label() == htree.sent_label():
                htree_bin = _htree
                break
        assert htree_bin is not None
        if not self.induction_settings.edge_labels:
            htree_bin.reset_edge_labels('--')
        grammar = direct_extract_lcfrs_from_prebinarized_corpus(htree_bin,
                                                                term_labeling=self.terminal_labeling,
                                                                nont_labeling=self.induction_settings.nont_labeling,
                                                                isolate_pos=self.induction_settings.isolate_pos,
                                                               )
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            grammar2 \
                = direct_extract_lcfrs_from_prebinarized_corpus(htree_bin,
                                                                term_labeling=self.terminal_labeling,
                                                                nont_labeling=self.induction_settings.nont_labeling,
                                                                isolate_pos=self.induction_settings.isolate_pos,
                                                               )
            self.terminal_labeling.backoff_mode = False
            grammar.add_gram(grammar2)

        if self.dummy_flag:
            print(grammar)
            self.dummy_flag = False

        return grammar, None

    def induce_from(self, obj):
        if self.induction_settings.use_discodop_binarization:
            return self.induce_from_disco_binarized(obj)

        if not self.__valid_tree(obj):
            print(obj, list(map(str, obj.token_yield())), obj.full_yield())
            return None, None
        if not self.induction_settings.edge_labels:
            obj.reset_edge_labels('--')
        grammar = direct_extract_lcfrs(obj,
                                       term_labeling=self.terminal_labeling,
                                       nont_labeling=self.induction_settings.nont_labeling,
                                       binarize=self.induction_settings.binarize,
                                       isolate_pos=self.induction_settings.isolate_pos,
                                       hmarkov=self.induction_settings.hmarkov)
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            grammar2 = direct_extract_lcfrs(obj,
                                            term_labeling=self.terminal_labeling,
                                            nont_labeling=self.induction_settings.nont_labeling,
                                            binarize=self.induction_settings.binarize,
                                            isolate_pos=self.induction_settings.isolate_pos,
                                            hmarkov=self.induction_settings.hmarkov)
            self.terminal_labeling.backoff_mode = False
            grammar.add_gram(grammar2)
        # print(grammar)
        # for rule in grammar.rules():
        #     print(rule)
        #     for lhs, rhs, dcp in binarize(rule.lhs(), rule.rhs(), rule.dcp()):
        #         bin_rule = LCFRS_rule(lhs=lhs, dcp=dcp)
        #         for rhs_nont in rhs:
        #             bin_rule.add_rhs_nont(rhs_nont)
        #         print("\t", bin_rule)
        # print()
        return grammar, None

    def initialize_parser(self):
        save_preprocess = (self.directory, "mygrammar")
        k = 1 if not self.organizer.disable_split_merge or self.oracle_parsing else self.k_best
        if "disco-dop" in self.parsing_mode:
            self.parser = DiscodopKbestParser(grammar=self.base_grammar, k=self.k_best,
                                              cfg_ctf=self.disco_dop_params["cfg_ctf"],
                                              pruning_k=self.disco_dop_params["pruning_k"],
                                              beam_beta=self.disco_dop_params["beam_beta"],
                                              beam_delta=self.disco_dop_params["beam_delta"]
                                             )
        else:
            self.parser = GFParser_k_best(self.base_grammar, save_preprocessing=save_preprocess, k=k)

    def compute_reducts(self, resource):

        # print_grammar(self.base_grammar)
        # for rule in self.base_grammar.rules():
        #     print(rule.get_idx(), rule)
        # sys.stdout.flush()

        training_corpus = list(filter(self.__valid_tree, self.read_corpus(resource)))

        if not self.induction_settings.edge_labels:
            for htree in training_corpus:
                htree.reset_edge_labels('--')

        parser = self.organizer.training_reducts.get_parser() if self.organizer.training_reducts is not None else None
        nonterminal_map = self.organizer.nonterminal_map
        frequency = self.backoff_factor if self.backoff else 1.0
        trace = compute_reducts(self.base_grammar, training_corpus, self.induction_settings.terminal_labeling,
                                parser=parser, nont_map=nonterminal_map, debug=False, frequency=frequency)
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            trace.compute_reducts(training_corpus, frequency=1.0)
            self.terminal_labeling.backoff_mode = False
        print("computed trace")
        return trace

    def print_config(self, file=None):
        if file is None:
            file = self.logger
        ConstituentExperiment.print_config(self, file=file)
        SplitMergeExperiment.print_config(self, file=file)

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

        SplitMergeExperiment.read_stage_file(self)


class Positional(object):
    def __init__(self, help='', type=None, choices=None, metavar=None):
        self.help = help + '. One of {' + ','.join(choices) + '}'
        self.kind = 'positional'
        self.abbrev = None
        self.type = type
        self.choices = None
        self.metavar = metavar


@plac.annotations(
    split=Positional('the corpus/split to run the experiment on', str, ["SPMRL", "HN08", "WSJ", "WSJ-km2003"]),
    test_mode=('evaluate on test set instead of dev. set', 'flag'),
    unk_threshold=('threshold for unking rare words', 'option', None, int),
    h_markov=('horizontal Markovization', 'option', None, int),
    v_markov=('vertical Markovization', 'option', None, int),
    no_edge_labels=('do not include edge labels in sDCP', 'flag'),
    quick=('run a small experiment (for testing/debugging)', 'flag'),
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
         unk_threshold=4,
         h_markov=1,
         v_markov=1,
         no_edge_labels=False,
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
         directory=None,
         counts_prior=0.0
         ):
    """
    Run an end-to-end experiment with LCFRS on constituent parsing.
    """
    induction_settings = InductionSettings()
    induction_settings.disconnect_punctuation = False
    induction_settings.normalize = True
    induction_settings.use_discodop_binarization = True
    binarization_settings = ["--headrules=" + ("util/negra.headrules" if split in ["SPMRL", "HN08"]
                                               else "util/ptb.headrules"),
                             "--binarize",
                             "-h " + str(h_markov),
                             "-v " + str(v_markov)]
    induction_settings.discodop_binarization_params = binarization_settings
    induction_settings.edge_labels = not no_edge_labels

    filters = []
    # filters += [check_single_child_label, lambda x: check_single_child_label(x, label="SB")]
    experiment = LCFRSExperiment(induction_settings, directory=directory, filters=filters)

    train, dev, test, test_input = setup_corpus_resources(split, not test_mode, quick, predicted_pos, TEST_SECOND_HALF)
    experiment.resources[TRAINING] = train
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
    experiment.counts_prior = counts_prior
    experiment.oracle_parsing = False
    experiment.k_best = k_best
    experiment.disco_dop_params["pruning_k"] = 50000
    experiment.read_stage_file()

    # only effective if no terminal labeling was read from stage file
    if experiment.terminal_labeling is None:
        # StanfordUNKing(experiment.read_corpus(experiment.resources[TRAINING]))
        experiment.set_terminal_labeling(terminal_labeling(experiment.read_corpus(experiment.resources[TRAINING]),
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
                                                       logger=experiment.logger,
                                                       secondary_scores=0)
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


__all__ = []
