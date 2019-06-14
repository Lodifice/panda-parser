from __future__ import print_function

from experiment.resources import TRAINING, TESTING, RESULT, Logger
from grammar.lcfrs import LCFRS
import tempfile
import multiprocessing
import os
import json
import pickle


class Experiment(object):
    def __init__(self, directory=None):
        print("Inititialize Experiment")
        if directory is None:
            import datetime
            now = datetime.datetime.now()
            self.directory = tempfile.mkdtemp(
                prefix='exp_' + '-'.join(
                    map(str, [now.year, now.month, now.day, '-', now.hour, now.minute, now.second])))
        else:
            self.directory = directory
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        self.logger = Logger(tempfile.mkstemp(dir=self.directory, suffix='.log')[1])
        self.stage_dict = {"stage": (0, )}
        self.base_grammar = None
        self.validator = None
        self.score_name = "score"
        self.parser = None
        self.result_file = None
        self.resources = {}  # to hold file paths, intervals of sentences etc.
        self.resources_data = {}  # to hold actual lists or sets
        self.parsing_timeout = None
        self.oracle_parsing = False
        self.max_score = None
        self.purge_rule_freq = None
        self.feature_log = None
        self.__stage_path = os.path.join(self.directory, "STAGEFILE")
        self.max_sentence_length_for_parsing = None

    @property
    def stage(self):
        return self.stage_dict["stage"]

    def read_stage_file(self):
        if os.path.exists(self.__stage_path):
            with open(self.__stage_path) as f:
                self.stage_dict = json.load(f)

                if "base_grammar" in self.stage_dict:
                    self.base_grammar = pickle.load(open(self.stage_dict["base_grammar"], 'rb'))

    def write_stage_file(self):
        with open(self.__stage_path, "w") as f:
            json.dump(self.stage_dict, f)

    def induction_preparation(self):
        """
        Hook to be called before grammar induction for setting up additional data sources
        or carrying out preprocessing steps.
        """
        pass

    def induce_grammar(self, corpus, start="START"):
        grammar = LCFRS(start=start)
        for obj in corpus:
            obj = self.preprocess_before_induction(obj)
            additional_induction_params = self.additional_induction_params(obj)
            obj_grammar, features = self.induce_from(obj, **additional_induction_params)
            if obj_grammar is None:
                continue
            if features is None:
                grammar.add_gram(obj_grammar, None)
            else:
                grammar.add_gram(obj_grammar, (self.feature_log, features))
        self.postprocess_grammar(grammar)
        self.base_grammar = grammar
        _, path = tempfile.mkstemp(suffix=".base.grammar", dir=self.directory)
        with open(path, 'wb') as f:
            pickle.dump(self.base_grammar, f)
            self.stage_dict["base_grammar"] = path

    def postprocess_grammar(self, grammar):
        if self.purge_rule_freq is not None:
            grammar.purge_rules(self.purge_rule_freq, self.feature_log)
        grammar.make_proper()

    def initialize_parser(self):
        assert False

    def preprocess_before_induction(self, obj):
        return obj

    def induce_from(self, obj, **kwargs):
        assert False

    def additional_induction_params(self, obj):
        return {}

    def parsing_preprocess(self, obj):
        assert False

    def parsing_postprocess(self, sentence, derivation, label=None):
        assert False

    def obtain_sentence(self, obj):
        assert False

    def obtain_label(self, obj):
        return None

    def score_object(self, obj, gold):
        return 0.0

    def mk_obj(self, args):
        assert False

    def do_parse(self, gold_corpus, test_inputs, result_resource):
        print("parsing, ", end='', file=self.logger)
        result_resource.init()
        from time import clock
        begin = clock()
        if self.parsing_timeout is None:
            for gold_obj, obj in zip(gold_corpus, test_inputs):
                parser_input = self.parsing_preprocess(obj)
                self.parser.set_input(parser_input)
                if self.max_sentence_length_for_parsing is None \
                        or self.max_sentence_length_for_parsing >= len(parser_input):
                    self.parser.parse()
                self.process_parse(gold_obj, obj, result_resource)
                self.parser.clear()
        else:
            self.do_parse_with_timeout(test_inputs, result_resource)
        print("\nParsing time, ", clock() - begin, file=self.logger)
        result_resource.finalize()

    def process_parse(self, gold_obj, obj, result_resource):
        sentence = self.obtain_sentence(obj)

        if self.parser.recognized():
            print(".", end='', file=self.logger)
            best_derivation = self.parser.best_derivation_tree()
            result = self.parsing_postprocess(sentence=sentence, derivation=best_derivation,
                                              label=self.obtain_label(obj))
        else:
            if self.max_sentence_length_for_parsing is None \
                    or len(self.parsing_preprocess(obj)) > self.max_sentence_length_for_parsing:
                print("s", end='', file=self.logger)
            else:
                print("-", end='', file=self.logger)
            result = self.compute_fallback(sentence=sentence)

        result_resource.write(self.serialize(result))

    def serialize(self, obj):
        assert False

    def run_experiment(self):
        self.print_config()

        # induction
        if self.stage[0] <= 1:
            training_corpus = self.read_corpus(self.resources[TRAINING])
            self.induction_preparation()
            self.induce_grammar(training_corpus)

        # weight training
        # omitted
        #

        if self.stage[0] <= 4:
            self.initialize_parser()

            # testing
            test_corpus = self.read_corpus(self.resources[TESTING])
            self.do_parse(test_corpus, test_corpus, self.resources[RESULT])

        if self.stage[0] <= 5:
            self.evaluate(self.resources[RESULT], self.resources[TESTING])

    def compute_fallback(self, sentence, label=None):
        assert False

    def read_corpus(self, resource):
        assert False

    def evaluate(self, result_resource, gold_resource):
        assert False

    def do_parse_with_timeout(self, test_inputs, result_resource):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        recognized = 0

        for obj in test_inputs:
            parser_input = self.parsing_preprocess(obj)
            self.parser.set_input(parser_input)

            timeout = False
            p = multiprocessing.Process(target=self.timeout_worker, args=(self.parser, obj, return_dict))
            p.start()
            p.join(timeout=self.parsing_timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                timeout = True

            if 0 in return_dict and return_dict[0] is not None:
                recognized += 1
                print(".", end='', file=self.logger)
                result = return_dict[0]
            else:
                if timeout:
                    print("t", end='', file=self.logger)
                else:
                    print("-", end='', file=self.logger)
                result = self.compute_fallback(self.obtain_sentence(obj), self.obtain_label(obj))

            self.post_parsing_action(obj, result, result_resource)

            return_dict[0] = None
            self.parser.clear()

        print(file=self.logger)
        print("From {} sentences, {} were recognized.".format(len(test_inputs), recognized), file=self.logger)

    def post_parsing_action(self, gold, system, result_resource):
        result_resource.write(self.serialize(system))

    def timeout_worker(self, parser, obj, return_dict):
        parser.parse()
        if parser.recognized():
            derivation = parser.best_derivation_tree()
            return_dict[0] = self.parsing_postprocess(sentence=self.obtain_sentence(obj), derivation=derivation
                                                      , label=self.obtain_label(obj))

    def compute_oracle_derivation(self, derivations, gold):
        best_der = None
        best_score = -1.0
        sentence = self.obtain_sentence(gold)
        label = self.obtain_label(gold)
        min_score = 1.0

        for derivation in derivations:
            system = self.parsing_postprocess(sentence, derivation, label)
            score = self.score_object(system, gold)
            # print(score, end=' ')
            if score > best_score:
                best_der, best_score = derivation, score
            if self.max_score is not None and best_score >= self.max_score:
                break
            if score < min_score:
                min_score = score
        print('max', best_score, 'min', min_score, file=self.logger)
        return best_der

    @staticmethod
    def precision_recall_f1(relevant, retrieved):
        inters = retrieved & relevant

        # in case of parse failure there are two options here:
        #   - parse failure -> no spans at all, thus precision = 1
        #   - parse failure -> a dummy tree with all spans wrong, thus precision = 0

        precision = 1.0 * len(inters) / len(retrieved) \
            if len(retrieved) > 0 else 0
        recall = 1.0 * len(inters) / len(relevant) \
            if len(relevant) > 0 else 0
        fmeasure = 2.0 * precision * recall / (precision + recall) \
            if precision + recall > 0 else 0

        return precision, recall, fmeasure

    def print_config(self, file=None):
        if file is None:
            file = self.logger
        print("Experiment infos for", self.__class__.__name__, file=file)
        print("Experiment base directory", self.directory, file=file)
        print("Purge rule freq:", self.purge_rule_freq, file=file)
        print("Max score", self.max_score, file=file)
        print("Score", self.score_name, file=file)
        print("Resources", '{\n' + '\n'.join(['\t' + str(k) + ' : ' + str(self.resources[k]) for k in self.resources]) + '\n}', file=file)
        print("Parsing Timeout: ", self.parsing_timeout, file=file)
        print("Oracle parsing: ", self.oracle_parsing, file=file)

    def update_stage(self, new_stage):
        if tuple(new_stage) > tuple(self.stage):
            self.stage_dict["stage"] = tuple(new_stage)


class ScoringExperiment(Experiment):
    def __init__(self, directory=None, filters=None):
        print("Initialize Scoring experiment")
        # super(ScoringExperiment, self).__init__()
        Experiment.__init__(self, directory=directory)
        self.filters = [] if filters is None else filters

    def process_parse(self, gold_obj, test_input, result_resource):
        sentence = self.obtain_sentence(test_input)

        if self.parser.recognized():
            print('.', end='', file=self.logger)
            self.logger.flush()
            if self.oracle_parsing:
                derivations = [der for _, der in self.parser.k_best_derivation_trees()]
                best_derivation = self.compute_oracle_derivation(derivations, gold_obj)
            else:
                best_derivation = self.parser.best_derivation_tree()
                if self.filters:
                    for _, der in self.parser.k_best_derivation_trees():
                        tree = self.parsing_postprocess(sentence=sentence, derivation=der,
                                                        label=self.obtain_label(gold_obj))
                        if all([predicate(tree) for predicate in self.filters]):
                            best_derivation = der
                            break

            secondaries = None
            if self.parser.secondaries:
                secondaries = []
                for mode in self.parser.secondaries:
                    self.parser.set_secondary_mode(mode)
                    der = self.parser.best_derivation_tree()
                    result = self.parsing_postprocess(sentence=sentence, derivation=der, label=self.obtain_label(gold_obj))
                    secondaries.append(result)
                self.parser.set_secondary_mode("DEFAULT")

            if best_derivation:
                result = self.parsing_postprocess(sentence=sentence, derivation=best_derivation,
                                              label=self.obtain_label(gold_obj))
                self.post_parsing_action(gold_obj, result, result_resource, secondaries)
            else:
                print('x', end='', file=self.logger)
                result_resource.failure(gold_obj)
        else:
            if self.max_sentence_length_for_parsing is not None and \
                    len(self.parsing_preprocess(test_input)) > self.max_sentence_length_for_parsing:
                print('s', end='', file=self.logger)
            else:
                print('-', end='', file=self.logger)
            result_resource.failure(gold_obj)

    def timeout_worker(self, parser, obj, return_dict):
        parser.parse()
        sentence = self.obtain_sentence(obj)

        if parser.recognized():
            if self.oracle_parsing:
                derivations = [der for _, der in self.parser.k_best_derivation_trees()]
                best_derivation = self.compute_oracle_derivation(derivations, obj)
            else:
                best_derivation = self.parser.best_derivation_tree()
                if self.filters:
                    for _, der in self.parser.k_best_derivation_trees():
                        tree = self.parsing_postprocess(sentence=sentence, derivation=der,
                                                        label=self.obtain_label(obj))
                        if all([predicate(tree) for predicate in self.filters]):
                            best_derivation = der
                            break

            result = self.parsing_postprocess(sentence=sentence, derivation=best_derivation,
                                                  label=self.obtain_label(obj))
            return_dict[0] = result

    def post_parsing_action(self, gold, system, result_resource, secondaries=None):
        result_resource.score(system, gold, secondaries)

    def activate_train_mode(self):
        pass

    def activate_test_mode(self):
        pass

__all__ = ["Experiment", "ScoringExperiment"]
