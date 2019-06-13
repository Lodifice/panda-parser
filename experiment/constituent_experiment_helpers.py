import os
from grammar.induction.terminal_labeling import  PosTerminals, TerminalLabeling, FeatureTerminals, \
    FrequencyBiasedTerminalLabeling, CompositionalTerminalLabeling, FormTerminals, deserialize_labeling, \
    FrequentSuffixTerminalLabeling, StanfordUNKing
from constituent.construct_morph_annotation import extract_feat
from experiment.resources import CorpusFile


SPLITS = ["SPMRL", "HN08", "WSJ", "WSJ-km2003", "negraall", "lassy-small"]


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
    #
    elif "WSJ" in split:
        # based on Kilian Evang's dptb.tar.bz2

        corpus_type = corpus_type_test = "EXPORT"
        corpus_path_original = "res/WSJ/ptb-discontinuous/dptb7.export"
        corpus_path_km2003 = "res/WSJ/ptb-discontinuous/dptb7-km2003wsj.export"

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
    elif split == "negraall":
        corpus_type = "EXPORT"
        train_start = validation_start = test_start = test_input_start = 1

        if quick:
            test_limit = validation_limit = test_input_limit = 200
            train_limit = 5000
        else:
            test_limit = validation_limit = test_input_limit = 1000
            train_limit = 18602
        validation_size = validation_limit
        train_path = "res/negraall/train.export"
        validation_path = "res/negraall/dev.export"
        if dev_mode:
            test_path = test_input_path = validation_path
        else:
            test_path = test_input_path = "res/negraall/test.export"
        corpus_type_test = 'EXPORT'
        train_exclude = validation_exclude = test_exclude = test_input_exclude = []
        train_filter = validation_filter = test_filter = test_input_filter = None

    elif split == "lassy-small":
        corpus_type = "EXPORT"
        train_start = validation_start = test_start = test_input_start = 1
        if quick:
            test_limit = validation_limit = test_input_limit = 200
            train_limit = 5000
        else:
            if dev_mode:
                test_limit = validation_limit = test_input_limit = 6520
            else:
                validation_limit = 6520
                test_limit = test_input_limit = 6523
            train_limit = 52157
        validation_size = validation_limit
        train_path = "res/lassy/lassytrain.norm.export"
        validation_path = "res/lassy/lassydev.norm.export"
        if dev_mode:
            test_input_path = test_path = validation_path
        else:
            test_input_path = test_path = "res/lassy/lassytest.norm.export"
        train_exclude = validation_exclude = test_exclude = test_input_exclude = []
        train_filter = validation_filter = test_filter = test_input_filter = None
        corpus_type_test = 'EXPORT'
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


def my_feature_filter(elem):
    base_feats = ["number", "person", "tense", "mood", "case", "degree", "category", "pos", "gender"]
    feat_set = {feat: value for feat, value in elem[0]}
    if "pos" in feat_set and feat_set["pos"] in {"APPR", "APPRART"}:
        return extract_feat(elem[0], features=base_feats + ["lemma"])
    return extract_feat(elem[0], features=base_feats)


# FINE_TERMINAL_LABELING = FeatureTerminals(token_to_features, feature_filter=my_feature_filter)
FINE_TERMINAL_LABELING = CompositionalTerminalLabeling(FormTerminals(), PosTerminals())
FALLBACK_TERMINAL_LABELING = PosTerminals()

DEFAULT_RARE_WORD_THRESHOLD = 10
TERMINAL_LABELINGS = ['form+pos', 'suffixes', 'suffixes+pos', 'stanford4']


def construct_terminal_labeling(labeling, corpus, threshold=DEFAULT_RARE_WORD_THRESHOLD):
    if labeling == "form+pos":
        return FrequencyBiasedTerminalLabeling(FINE_TERMINAL_LABELING, FALLBACK_TERMINAL_LABELING, corpus, threshold)
    elif labeling == "suffixes":
        return FrequentSuffixTerminalLabeling(corpus, threshold)
    elif labeling == 'suffixes+pos':
        return CompositionalTerminalLabeling(FrequentSuffixTerminalLabeling(corpus, threshold), PosTerminals())
    elif labeling == 'stanford4':
        return StanfordUNKing(corpus, unknown_threshold=threshold, openclass_threshold=50)
    else:
        raise Exception("Unknown terminal labeling \"%s\"" % labeling)


__all__ = ['setup_corpus_resources', 'SPLITS', 'construct_terminal_labeling', 'TERMINAL_LABELINGS']
