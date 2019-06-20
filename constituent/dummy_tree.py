from hybridtree.constituent_tree import ConstituentTree, ConstituentCategory


def dummy_constituent_tree(token_yield, full_token_yield, dummy_label, dummy_root, label=None):
    """
    :param token_yield: connected yield of a parse tree
    :type token_yield: list[ConstituentTerminal]
    :param full_token_yield: full yield of the parse tree
    :type full_token_yield: list[ConstituentTerminal]
    :return: dummy constituent tree
    :rtype: ConstituentTree
    generates a dummy tree for a given yield using dummy_label as inner node symbol
    """
    tree = ConstituentTree(label)

    # create all leaves and punctuation
    for token in full_token_yield:
        if token not in token_yield:
            tree.add_punct(full_token_yield.index(token), token.pos(), token.form())
        else:
            tree.add_leaf(full_token_yield.index(token), token.pos(), token.form())

    # generate root node
    root_id = 'n0'
    tree.add_node(root_id, ConstituentCategory(dummy_root))
    tree.add_to_root(root_id)

    parent = root_id

    if len(token_yield) > 1:
        i = 1
        # generate inner nodes of branching tree
        for token in token_yield[:-2]:
            node = ConstituentCategory(str(dummy_label))
            tree.add_node('n' + str(i), node)
            tree.add_child(parent, 'n' + str(i))
            tree.add_child(parent, full_token_yield.index(token))
            parent = 'n' + str(i)
            i += 1

        token = token_yield[len(token_yield) - 2]
        tree.add_child(parent, full_token_yield.index(token))
        token = token_yield[len(token_yield) - 1]
        tree.add_child(parent, full_token_yield.index(token))
    elif len(token_yield) == 1:
        tree.add_child(parent, full_token_yield.index(token_yield[0]))

    return tree


def flat_dummy_constituent_tree(token_yield, full_token_yield, dummy_label, dummy_root, label=None, gold_pos=True):
    """
    :param token_yield: connected yield of a parse tree
    :type token_yield: list[ConstituentTerminal]
    :param full_token_yield: full yield of the parse tree
    :type full_token_yield: list[ConstituentTerminal]
    :return: dummy constituent tree
    :rtype: ConstituentTree
    generates a flat dummy tree for a given yield where all nodes are attached under the root
    """
    tree = ConstituentTree(label)

    # generate root node
    root_id = 'n_root'
    tree.add_node(root_id, ConstituentCategory(dummy_root))
    tree.add_to_root(root_id)

    parent = root_id

    # create all leaves and punctuation
    for token in full_token_yield:
        pos = token.pos() if gold_pos else '--'
        if token not in token_yield:
            tree.add_punct(full_token_yield.index(token), pos, token.form())
        else:
            idx = full_token_yield.index(token)
            tree.add_leaf(idx, pos, token.form(), morph=token.morph_feats(), lemma=token.lemma())

            tree.add_child(parent, idx)

    return tree
