from parser.naive.parsing import LCFRS_parser as NaiveParser
from parser.viterbi.viterbi import ViterbiParser, LeftCornerParser, RightBranchingParser
from parser.viterbi.left_branching import LeftBranchingParser
import re

class ParserFactory:
    def __init__(self):
        self.__parsers = {}

    def registerParser(self, name, parser):
        self.__parsers[name] = parser

    def getParser(self, name):
        match = re.search(r'fanout-(\d+)', name)
        if match:
            return ViterbiParser
        return self.__parsers[name]


def the_parser_factory():
    factory = ParserFactory()
    factory.registerParser('left-branching', LeftBranchingParser)
    factory.registerParser('right-branching', RightBranchingParser)
    factory.registerParser('direct-extraction', ViterbiParser)
    factory.registerParser('viterbi-bottom-up', ViterbiParser)
    factory.registerParser('naive-bottom-up', NaiveParser)
    factory.registerParser('viterbi-left-corner', LeftCornerParser)
    factory.registerParser('earley-left-to-right', RightBranchingParser)
    return factory