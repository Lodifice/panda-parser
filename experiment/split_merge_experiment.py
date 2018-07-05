import multiprocessing
import pickle
import tempfile

from experiment.base_experiment import Experiment
from experiment.resources import TRAINING, VALIDATION, TESTING, TESTING_INPUT, RESULT
from parser.coarse_to_fine_parser.coarse_to_fine import Coarse_to_fine_parser
from parser.discodop_parser.parser import DiscodopKbestParser
try:
    from parser.gf_parser.gf_interface import GFParser_k_best
except ImportError:
    print("The Grammatical Framework is not installed properly – the GFParser is unavailable.")
from parser.lcfrs_la import build_sm_grammar
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.trace_manager.score_validator import PyCandidateScoreValidator
from parser.trace_manager.sm_trainer import PySplitMergeTrainerBuilder, build_PyLatentAnnotation_initial, \
    build_PyLatentAnnotation
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager


class SplitMergeOrganizer:
    def __init__(self):
        # basic objects
        self.grammarInfo = None
        self.storageManager = None
        self.nonterminal_map = None

        # settings and training ingredients
        self.disable_em = False
        self.disable_split_merge = False
        self.training_reducts = None
        self.em_epochs = 20
        self.em_epochs_sm = 20
        self.max_sm_cycles = 2
        self.min_epochs = 6
        self.min_epochs_smoothing = 3
        self.ignore_failures_smoothing = False
        self.threads = 1
        self.seed = 0
        self.validationDropIterations = 6
        self.smoothing_factor = 0.01
        self.smoothing_factor_unary = 0.1
        self.split_randomization = 1.0  # in percent
        self.validator_type = "SCORE"  # SCORE or SIMPLE
        self.validator = None  # required for SCORE validation
        self.refresh_score_validator = False  # rebuild the k-best candidate list after each split/merge cycle
        self.project_weights_before_parsing = True
        self.validation_reducts = None  # required for SIMPLE validation
        self.merge_percentage = 50.0
        self.merge_type = "PERCENT" # or SCC or THRESHOLD
        self.merge_threshold = -0.2
        self.merge_interpolation_factor = 0.9

        # the trainer state
        self.splitMergeTrainer = None
        self.emTrainer = None
        self.latent_annotations = {}
        self.merge_sources = {}
        self.last_sm_cycle = None

    def __str__(self):
        s = "Split/Merge Settings {\n"
        for key in self.__dict__:
            if not key.startswith("__") and key not in ["grammarInfo", "storageManager", "nonterminal_map"]:
                s += "\t" + key + ": " + str(self.__dict__[key]) + "\n"
        return s + "}\n"


class SplitMergeExperiment(Experiment):
    def __init__(self):
        print("Initialize Split Merge Experiment", file=self.logger)
        self.organizer = SplitMergeOrganizer()
        self.parsing_mode = "k-best-rerank"  # or "best-latent-derivation"
        self.k_best = 50
        self.heuristics = -1.0
        self.disco_dop_params = {"beam_beta": 0.0,
                                 "beam_delta": 50,
                                 "pruning_k": 10000,
                                 "cfg_ctf": True}

    def read_stage_file(self):
        # super(SplitMergeExperiment, self).read_stage_file()
        if "latent_annotations" in self.stage_dict:
            # this is a workaround
            if self.organizer.training_reducts is None:
                self.update_reducts(self.compute_reducts(self.resources[TRAINING]), type=TRAINING)
                self.write_stage_file()
            # this was a workaround
            self.initialize_training_environment()

            las = self.stage_dict["latent_annotations"]
            for key in las:
                with open(las[key], "rb") as f:
                    splits, rootWeights, ruleWeights = pickle.load(f)
                    # print(key)
                    # print(len(splits), len(rootWeights), len(ruleWeights))
                    # print(len(self.base_grammar.nonts()))
                    la = build_PyLatentAnnotation(splits, rootWeights, ruleWeights, self.organizer.grammarInfo, self.organizer.storageManager)
                    self.organizer.latent_annotations[int(key)] = la
        if "last_sm_cycle" in self.stage_dict:
            self.organizer.last_sm_cycle = int(self.stage_dict["last_sm_cycle"])
            # TODO: delete unused latent annotations

    def build_score_validator(self, resource):
        self.organizer.validator = PyCandidateScoreValidator(self.organizer.grammarInfo
                                                             , self.organizer.storageManager, self.score_name)

        corpus_validation = self.read_corpus(resource)
        obj_count = 0
        der_count = 0
        timeout = False

        if self.parsing_timeout:
            timeout_manager = multiprocessing.Manager()
            return_dict = timeout_manager.dict()

        for gold in corpus_validation:
            obj_count += 1
            self.parser.set_input(self.parsing_preprocess(gold))

            if self.parsing_timeout:
                timeout, derivations_ = self._compute_derivations_with_timeout(return_dict)
                derivations = list(map(lambda x: x[1], derivations_))
            else:
                self.parser.parse()
                derivations = list(map(lambda x: x[1], self.parser.k_best_derivation_trees()))

            manager = PyDerivationManager(self.base_grammar, self.organizer.nonterminal_map)
            manager.convert_derivations_to_hypergraphs(derivations)
            scores = []

            # derivations = self.parser.k_best_derivation_trees()
            for der in derivations:
                der_count += 1
                result = self.parsing_postprocess(self.obtain_sentence(gold), der)
                score = self.score_object(result, gold)
                scores.append(score)

            self.organizer.validator.add_scored_candidates(manager, scores, self.max_score)
            # print(obj_count, self.max_score, scores)
            token = 't' if timeout else ('.' if scores else '-')
            print(token, end='', file=self.logger)
            if scores:
                print(obj_count, 'max', max(scores), 'firsts', scores[0:10], file=self.logger)
            else:
                print(obj_count, 'max 00.00', '[]', file=self.logger)
            self.parser.clear()
        # print("trees used for validation ", obj_count, "with", der_count * 1.0 / obj_count, "derivations on average")

    def _compute_derivations_with_timeout(self, return_dict):
        p = multiprocessing.Process(target=self._derivations_timeout_worker, args=(self.parser, return_dict))
        p.start()
        p.join(timeout=self.parsing_timeout)
        if p.is_alive():
            p.terminate()
            p.join()
        if 0 in return_dict and return_dict[0] is not None:
            result = return_dict[0]
            # return_dict[0] = None
            return False, result
        return True, []

    def _derivations_timeout_worker(self, parser, return_dict):
        return_dict[0] = None
        parser.parse()
        if parser.recognized():
            return_dict[0] = list(parser.k_best_derivation_trees())

    def compute_reducts(self, resource):
        assert False

    def update_reducts(self, trace, type=TRAINING):
        _, trace_path = tempfile.mkstemp("." + type + ".reduct", dir=self.directory)
        trace.serialize(bytes(trace_path, encoding="utf-8"))
        print("Serialized " + type + " reducts to " + trace_path, file=self.logger)
        if type is TRAINING:
            self.organizer.training_reducts = trace
            self.stage_dict["training_reducts"] = trace_path
        elif type is VALIDATION:
            self.organizer.validation_reducts = trace
            self.stage_dict["validation_reducts"] = trace_path

    def do_em_training(self):
        em_builder = PySplitMergeTrainerBuilder(self.organizer.training_reducts, self.organizer.grammarInfo)
        em_builder.set_em_epochs(self.organizer.em_epochs)
        em_builder.set_simple_expector(threads=self.organizer.threads)
        em_builder.set_scc_merger(self.organizer.merge_threshold)
        em_builder.set_scc_merge_threshold_function(self.organizer.merge_interpolation_factor)
        self.organizer.emTrainer = emTrainer = em_builder.build()

        initial_la = self.create_initial_la()

        emTrainer.em_train(initial_la)
        try:
            initial_la.project_weights(self.base_grammar, self.organizer.grammarInfo)
        except Exception as exc:
            nont_idx = exc.args[0]
            splits, root_weights, rule_weights = initial_la.serialize()
            nont = self.organizer.nonterminal_map.index_object(nont_idx)
            print(nont, nont_idx, splits[nont_idx], file=self.logger)
            for rule in self.base_grammar.lhs_nont_to_rules(nont):
                print(rule, rule_weights[rule.get_idx()], file=self.logger)
            raise

        self.organizer.latent_annotations[0] = initial_la
        self.organizer.last_sm_cycle = 0
        self.save_current_la()

    def save_current_la(self):
        cycle = self.stage_dict["last_sm_cycle"] = self.organizer.last_sm_cycle
        _, la_path = tempfile.mkstemp(suffix=".la" + str(cycle) + ".pkl", dir=self.directory)
        with open(la_path, 'wb') as f:
            pickle.dump(self.organizer.latent_annotations[cycle].serialize(), f)
            if "latent_annotations" not in self.stage_dict:
                self.stage_dict["latent_annotations"] = {}
            self.stage_dict["latent_annotations"][cycle] = la_path

    def create_initial_la(self):
        # randomize initial weights and do em training
        la_no_splits = build_PyLatentAnnotation_initial(self.base_grammar, self.organizer.grammarInfo,
                                                        self.organizer.storageManager)
        la_no_splits.add_random_noise(seed=self.organizer.seed)
        return la_no_splits

    def prepare_split_merge_trainer(self):
        # prepare SM training
        builder = PySplitMergeTrainerBuilder(self.organizer.training_reducts, self.organizer.grammarInfo)
        builder.set_em_epochs(self.organizer.em_epochs_sm)
        builder.set_simple_expector(threads=self.organizer.threads)
        if self.organizer.validator_type == "SCORE":
            builder.set_score_validator(self.organizer.validator, self.organizer.validationDropIterations)
        elif self.organizer.validator_type == "SIMPLE":
            builder.set_simple_validator(self.organizer.validation_reducts, self.organizer.validationDropIterations)
        builder.set_smoothing_factor(smoothingFactor=self.organizer.smoothing_factor,
                                     smoothingFactorUnary=self.organizer.smoothing_factor_unary)
        builder.set_split_randomization(percent=self.organizer.split_randomization, seed=self.organizer.seed + 1)

        # set merger
        if self.organizer.merge_type == "SCC":
            builder.set_scc_merger(self.organizer.merge_threshold)
        elif self.organizer.merge_type == "THRESHOLD":
            builder.set_threshold_merger(self.organizer.merge_threshold)
        else:
            builder.set_percent_merger(self.organizer.merge_percentage)

        self.custom_sm_options(builder)
        self.organizer.splitMergeTrainer = builder.build()

        if self.organizer.validator_type in ["SCORE", "SIMPLE"]:
            self.organizer.splitMergeTrainer.setMaxDrops(self.organizer.validationDropIterations, mode="smoothing")
            self.organizer.splitMergeTrainer.setMinEpochs(self.organizer.min_epochs)
            self.organizer.splitMergeTrainer.setMinEpochs(self.organizer.min_epochs_smoothing, mode="smoothing")
            self.organizer.splitMergeTrainer.setIgnoreFailures(self.organizer.ignore_failures_smoothing, mode="smoothing")
        self.organizer.splitMergeTrainer.setEMepochs(self.organizer.em_epochs_sm, mode="smoothing")

    def custom_sm_options(self, builder):
        pass

    def run_split_merge_cycle(self):
        if self.organizer.last_sm_cycle is None:
            la_no_splits = self.create_initial_la()
            self.organizer.last_sm_cycle = 0
            self.organizer.latent_annotations[0] = la_no_splits
            self.save_current_la()

        current_la = self.organizer.latent_annotations[self.organizer.last_sm_cycle]
        next_la = self.organizer.splitMergeTrainer.split_merge_cycle(current_la)
        next_cycle = self.organizer.last_sm_cycle + 1
        self.organizer.last_sm_cycle = next_cycle
        self.organizer.latent_annotations[next_cycle] = next_la
        self.organizer.merge_sources[next_cycle] = self.organizer.splitMergeTrainer.get_current_merge_sources()
        self.save_current_la()

    def initialize_training_environment(self):
        self.organizer.nonterminal_map = self.organizer.training_reducts.get_nonterminal_map()
        self.organizer.grammarInfo = PyGrammarInfo(self.base_grammar, self.organizer.nonterminal_map)
        self.organizer.storageManager = PyStorageManager()

    def prepare_sm_parser(self):
        last_la = self.organizer.latent_annotations[self.organizer.last_sm_cycle]
        if self.parsing_mode == "discodop-multi-method":
            if self.organizer.project_weights_before_parsing:
                self.project_weights()
            self.parser = DiscodopKbestParser(self.base_grammar,
                                              k=self.k_best,
                                              la=last_la,
                                              nontMap=self.organizer.nonterminal_map,
                                              variational=False,
                                              sum_op=False,
                                              cfg_ctf=self.disco_dop_params["cfg_ctf"],
                                              beam_beta=self.disco_dop_params["beam_beta"],
                                              beam_delta=self.disco_dop_params["beam_delta"],
                                              pruning_k=self.disco_dop_params["pruning_k"],
                                              grammarInfo=self.organizer.grammarInfo,
                                              projection_mode=False,
                                              latent_viterbi_mode=True,
                                              secondaries=["VARIATIONAL", "MAX-RULE-PRODUCT", "LATENT-RERANK"]
            )
            self.parser.k_best_reranker = Coarse_to_fine_parser(self.base_grammar, last_la,
                                                                self.organizer.grammarInfo,
                                                                self.organizer.nonterminal_map,
                                                                base_parser=self.parser)

        elif self.parsing_mode == "best-latent-derivation":
            grammar = build_sm_grammar(last_la,
                                       self.base_grammar,
                                       self.organizer.grammarInfo,
                                       rule_pruning=0.0001,
                                       rule_smoothing=0.1)
            self.parser = GFParser_k_best(grammar=grammar, k=1, save_preprocessing=(self.directory, "gfgrammar"))
        elif self.parsing_mode in { method + engine
                                    for method in {"k-best-rerank", "latent-viterbi"}
                                    for engine in {"-GF", "-disco-dop", ""}
                                  }:
            if self.organizer.project_weights_before_parsing:
                self.project_weights()
            if "disco-dop" in self.parsing_mode:
                engine = DiscodopKbestParser(grammar=self.base_grammar,
                                             k=self.k_best,
                                             la=last_la,
                                             nontMap=self.organizer.nonterminal_map,
                                             grammarInfo=self.organizer.grammarInfo,
                                             cfg_ctf=self.disco_dop_params["cfg_ctf"],
                                             beam_beta=self.disco_dop_params["beam_beta"],
                                             beam_delta=self.disco_dop_params["beam_beta"],
                                             pruning_k=self.disco_dop_params["pruning_k"],
                                             latent_viterbi_mode="latent-viterbi" in self.parsing_mode
                                             )
            else:
                engine = GFParser_k_best(grammar=self.base_grammar, k=self.k_best, heuristics=self.heuristics, save_preprocessing=(self.directory, "gfgrammar"))
            if "latent-viterbi" in self.parsing_mode:
                self.parser = engine
            else:
                self.parser = Coarse_to_fine_parser(self.base_grammar,                                  last_la,
                                                    self.organizer.grammarInfo,
                                                    self.organizer.nonterminal_map,
                                                    base_parser=engine)
        elif self.parsing_mode in {method + "%s" % engine
                                   for method in {"max-rule-prod", "max-rule-sum", "variational"}
                                   for engine in {"-GF", "-disco-dop", ""}}:
            if self.organizer.project_weights_before_parsing:
                self.project_weights()
            if "GF" in self.parsing_mode:
                self.parser = Coarse_to_fine_parser(self.base_grammar,
                                                    last_la,
                                                    self.organizer.grammarInfo,
                                                    nontMap=self.organizer.nonterminal_map,
                                                    base_parser_type=GFParser_k_best,
                                                    k=self.k_best,
                                                    heuristics=self.heuristics,
                                                    save_preprocessing=(self.directory, "gfgrammar"),
                                                    mode=self.parsing_mode,
                                                    variational="variational" in self.parsing_mode,
                                                    sum_op="sum" in self.parsing_mode)
            else:
                self.parser = DiscodopKbestParser(self.base_grammar,
                                                  k=self.k_best,
                                                  la=last_la,
                                                  nontMap=self.organizer.nonterminal_map,
                                                  variational="variational" in self.parsing_mode,
                                                  sum_op="sum" in self.parsing_mode,
                                                  cfg_ctf=self.disco_dop_params["cfg_ctf"],
                                                  beam_beta=self.disco_dop_params["beam_beta"],
                                                  beam_delta=self.disco_dop_params["beam_delta"],
                                                  pruning_k=self.disco_dop_params["pruning_k"],
                                                  grammarInfo=self.organizer.grammarInfo,
                                                  projection_mode=True)

        else:
            raise ValueError("Unknown parsing mode %s" % self.parsing_mode)

    def project_weights(self):
        last_la = self.organizer.latent_annotations[self.organizer.last_sm_cycle]

        if True:
            last_la.project_weights(self.base_grammar, self.organizer.grammarInfo)
        else:
            splits, _, _ = last_la.serialize()
            merge_sources = [[[split for split in range(0, splits[nont_idx])]]
                             for nont_idx in range(0, self.organizer.nonterminal_map.get_counter())]

            # print("Projecting to fine grammar LA", file=self.logger)
            coarse_la = last_la.project_annotation_by_merging(self.organizer.grammarInfo, merge_sources)
            coarse_la.project_weights(self.base_grammar, self.organizer.grammarInfo)

    def run_experiment(self):
        self.print_config()

        # induction
        if self.stage[0] <= 1:
            training_corpus = self.read_corpus(self.resources[TRAINING])
            self.induction_preparation()
            self.induce_grammar(training_corpus)

        if self.stage[0] <= 2:
            # prepare reducts
            if not self.organizer.disable_split_merge or not self.organizer.disable_em:
                self.update_reducts(self.compute_reducts(self.resources[TRAINING]), type=TRAINING)
                self.initialize_training_environment()

            # create initial LA and do EM training
            if not self.organizer.disable_em:
                self.do_em_training()
                self.update_stage((3,))
                self.write_stage_file()

        if self.stage[0] <= 3:
            if not self.organizer.disable_split_merge:
                if self.organizer.validator_type == "SCORE" and self.organizer.validator is None:
                    self.initialize_parser()
                    self.build_score_validator(self.resources[VALIDATION])
                elif self.organizer.validator_type == "SIMPLE" and self.organizer.validation_reducts is None:
                    self.update_reducts(self.compute_reducts(self.resources[VALIDATION]), type=VALIDATION)
                self.prepare_split_merge_trainer()

                while self.organizer.last_sm_cycle is None \
                        or self.organizer.last_sm_cycle < self.organizer.max_sm_cycles:
                    self.run_split_merge_cycle()
                    if self.organizer.last_sm_cycle < self.organizer.max_sm_cycles \
                            and self.organizer.validator_type == "SCORE" \
                            and self.organizer.refresh_score_validator:
                        self.project_weights()
                        self.initialize_parser()
                        self.build_score_validator(self.resources[VALIDATION])
                    self.write_stage_file()

        if self.stage[0] <= 4:
            if self.organizer.disable_split_merge and self.organizer.disable_em:
                self.initialize_parser()
            else:
                self.prepare_sm_parser()

            # testing
            if self.parsing_mode == "discodop-multi-method":
                print(self.parser.secondaries)
                assert self.parser.secondaries is not None
            test_gold = self.read_corpus(self.resources[TESTING])
            test_input = self.read_corpus(self.resources[TESTING_INPUT])
            self.do_parse(test_gold, test_input, self.resources[RESULT])

        if self.stage[0] <= 5:
            self.evaluate(self.resources[RESULT], self.resources[TESTING])

    def print_config(self, file=None):
        if file is None:
            file = self.logger
        print(self.organizer, file=file)
        print("Split/Merge Parsing mode: ", self.parsing_mode, file=file)
        print("k-best", self.k_best)
        print("heuristics", self.heuristics)
        print("disco-dop engine settings", self.disco_dop_params, file=file)


__all__ = ["SplitMergeExperiment", "SplitMergeOrganizer"]
