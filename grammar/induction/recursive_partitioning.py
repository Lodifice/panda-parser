import re
from grammar.induction.decomposition import left_branching_partitioning, right_branching_partitioning, fanout_limited_partitioning, fanout_limited_partitioning_left_to_right, fanout_limited_partitioning_argmax, fanout_limited_partitioning_random_choice, fanout_limited_partitioning_no_new_nont, fanout_limit_partitioning_with_guided_binarization
from random import seed


# Recursive partitioning strategies

def left_branching(tree):
    return left_branching_partitioning(len(tree.id_yield()))


def right_branching(tree):
    return right_branching_partitioning(len(tree.id_yield()))


def direct_extraction(tree):
    return tree.recursive_partitioning()


def cfg(tree):
    return fanout_k(tree, 1)


fanout_k = lambda tree, k: fanout_limited_partitioning(tree.recursive_partitioning(), k)
fanout_k_left_to_right = lambda tree, k: fanout_limited_partitioning_left_to_right(tree.recursive_partitioning(), k)
fanout_k_argmax = lambda tree, k: fanout_limited_partitioning_argmax(tree.recursive_partitioning(), k)
fanout_k_random = lambda tree, k: fanout_limited_partitioning_random_choice(tree.recursive_partitioning(), k)
fanout_k_no_new_nont = lambda tree, nonts, nont_labelling, fallback, k: fanout_limited_partitioning_no_new_nont(tree.recursive_partitioning(), k, tree, nonts, nont_labelling, fallback)


class RecursivePartitioningFactory:
    def __init__(self):
        self.__partitionings = {}

    def register_partitioning(self, name, partitioning):
        self.__partitionings[name] = partitioning

    def get_partitioning(self, name):
        partitioning_names = name.split(',')
        partitionings = []
        for name in partitioning_names:
            match = re.search(r'fanout-(\d+)([-\w]*)', name)
            if match:
                rec_par = None
                k = int(match.group(1))
                trans = match.group(2)
                if trans == '': #right-to-left bfs
                    rec_par = lambda tree: fanout_k(tree, k)
                    rec_par.__name__ = 'fanout_' + str(k)
                elif trans == '-left-to-right':
                    rec_par = lambda tree: fanout_k_left_to_right(tree, k)
                    rec_par.__name__ = 'fanout_' + str(k) + '_left_to_right'
                elif trans == '-argmax':
                    rec_par = lambda tree: fanout_k_argmax(tree, k)
                    rec_par.__name__ = 'fanout_' + str(k) + '_argmax'
                    partitionings.append(rec_par)
                elif trans == '-guided-binarization':
                    rec_par = lambda tree, reference_tree: \
                        fanout_limit_partitioning_with_guided_binarization(tree.recursive_partitioning(),
                                                                           k,
                                                                           reference_tree)
                    rec_par.__name__ = 'fanout_' + str(k) + '_guided_binarization'
                else:
                    randMatch = re.search(r'-random-(\d*)', trans)
                    if randMatch:
                        # set seed, if random strategy is chosen
                        s = int(randMatch.group(1))
                        seed(s)
                        rec_par = lambda tree: fanout_k_random(tree, k)
                        rec_par.__name__ = 'fanout_' + str(k) + '_random'

                    noNewMatch = re.search(r'-no-new-nont([-\w]*)', trans)
                    if noNewMatch:
                        # set fallback strategy if no position corresponds to an existing nonterminal
                        fallback = noNewMatch.group(1)
                        randMatch = re.search(r'-random-(\d*)', fallback)
                        if randMatch:
                            s = int(randMatch.group(1))
                            seed(s)
                            fallback = '-random'
                        rec_par = lambda tree, nonts, nont_labelling: fanout_k_no_new_nont(tree, nonts, nont_labelling, k, fallback)
                        rec_par.__name__ = 'fanout_' + str(k) + '_no_new_nont'
                if rec_par is not None:
                    partitionings.append(rec_par)
            else:
                rec_par = self.__partitionings[name]
                if rec_par:
                    partitionings.append(rec_par)
                else:
                    return None
        if partitionings:
            return partitionings
        else:
            return None


def the_recursive_partitioning_factory():
    factory = RecursivePartitioningFactory()
    factory.register_partitioning('left-branching', left_branching)
    factory.register_partitioning('right-branching', right_branching)
    factory.register_partitioning('direct-extraction', direct_extraction)
    factory.register_partitioning('cfg', cfg)
    return factory


__all__ = ["RecursivePartitioningFactory", "left_branching", "right_branching", "cfg", "direct_extraction"]
