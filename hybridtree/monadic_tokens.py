#-*- coding: utf-8 -*-
__author__ = 'kilian'

from abc import ABCMeta, abstractmethod


class MonadicToken(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rank(self):
        """
        :rtype: int
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        :rtype: str
        """
        pass

    @abstractmethod
    def type(self):
        pass


class CoNLLToken(MonadicToken):
    def __init__(self, form, lemma, cpos, pos, feats, deprel):
        super(CoNLLToken, self).__init__()
        self.__form = form
        self.__lemma = lemma
        self.__cpos = cpos
        self.__pos = pos
        self.__feats = feats
        self.__deprel = deprel

    def rank(self):
        return 1

    def form(self):
        return self.__form

    def lemma(self):
        return self.__lemma

    def cpos(self):
        return self.__cpos

    def pos(self):
        return self.__pos

    def feats(self):
        return self.__feats

    def deprel(self):
        return self.__deprel

    def set_edge_label(self, deprel):
        self.__deprel = deprel

    def set_pos_tag(self, pos: str):
        self.__pos = pos

    def __str__(self):
        return self.form() + ' : ' + self.pos() + ' : ' + self.deprel()

    def __eq__(self, other):
        if not isinstance(other, CoNLLToken):
            return False
        else:
            return all([self.form() == other.form()
                           , self.cpos() == other.cpos()
                           , self.pos() == other.pos()
                           , self.feats() == other.feats()
                           , self.lemma() == other.lemma()
                           , self.deprel() == other.deprel()])

    def __hash__(self):
        return hash((self.__form, self.__cpos, self.__pos, self.__feats, self.__lemma, self.__deprel))

    def type(self):
        return "CONLL-X"


class ConstituencyToken(MonadicToken):
    def __init__(self):
        super(ConstituencyToken, self).__init__()
        self._edge = None

    @abstractmethod
    def rank(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def edge(self):
        return self._edge

    def set_edge_label(self, edge):
        self._edge = edge


class ConstituentTerminal(ConstituencyToken):
    def __init__(self, form, pos, edge='--', morph=None, lemma='--'):
        super(ConstituentTerminal, self).__init__()
        self._edge = edge
        self.__form = form
        self.__pos = pos
        self._morph = [] if morph is None else morph
        self.__lemma = lemma

    def rank(self):
        return 0

    def lemma(self):
        return self.__lemma

    def form(self):
        return self.__form

    def pos(self):
        return self.__pos

    def morph_feats(self):
        return self._morph

    def __str__(self):
        return self.form() + "[" + self.__lemma + "]" + ' : ' + self.pos() + '\t' + str(self.edge())\
                   + '\t' + str(self._morph)
        # try:
        #     return self.form().encode("utf_8") + "[" + self.__lemma + "]"+ ' : ' + self.pos() + '\t' + str(self.edge())\
        #            + '\t' + str(self._morph)
        # except UnicodeDecodeError:
        #     return ' : ' + self.pos() + '\t' + str(self.edge()) + '\t' + str(self._morph)

    def __hash__(self):
        return hash((self.__form, self.__pos))

    def type(self):
        return "CONSTITUENT-TERMINAL"

    def set_pos_tag(self, pos: str):
        self.__pos = pos


class ConstituentCategory(ConstituencyToken):
    def __init__(self, category, edge='--'):
        super(ConstituentCategory, self).__init__()
        self.__category = category
        self._edge = edge

    def rank(self):
        return 1

    def category(self):
        return self.__category

    def __str__(self):
        return str(self.category()) + '\t' + self.edge()

    def __hash__(self):
        return hash((self.__category, self._edge))

    def type(self):
        return "CONSTITUENT-CATEGORY"

    def set_category(self, category):
        self.__category = category


def construct_conll_token(form, pos, _=True):
    return CoNLLToken(form, '_', pos, pos, '_', '_')


def construct_constituent_token(form, pos, terminal):
    if terminal:
        return ConstituentTerminal(form, pos)
    elif isinstance(form, str):
        return ConstituentCategory(form)
    else:
        return ConstituentCategory(form.get_string(), edge=form.edge_label())


__all__ = ["ConstituencyToken", "ConstituentTerminal", "ConstituentCategory", "CoNLLToken", "construct_conll_token",
           "construct_constituent_token"]
