import os, sys, plac, json, pickle
from grammar.lcfrs import *
from parser.sDCP_parser.sdcp_trace_manager import PySDCPTraceManager
from grammar.induction.terminal_labeling import deserialize_labeling


@plac.annotations(
    directory=('directory in which experiment is run (default: .)', 'option', 'd', str),
    posinfo=('comma-separated list of POS to show statistics on (default: "")', 'option', None, str),
    showpos=('show list of POS tags', 'flag'),
    allposinfo=('show statistics for all POS tags', 'flag'),
    number=('number of lexical entries per POS tag', 'option', 'n', int)
)
def main(directory: str = '.', posinfo: str = '', showpos=False, allposinfo=False, number=1):
    rootpath = os.path.abspath(directory)
    def changepath(path):
        return os.path.join(rootpath, os.path.split(path)[1])

    stage_file_path = os.path.join(rootpath, 'STAGEFILE')
    if os.path.isfile(stage_file_path):
        with open(stage_file_path) as stage_file:
            stage_dict = json.load(stage_file)
            print(stage_dict)

        gr_path = changepath(stage_dict['base_grammar'])

        with open(gr_path, 'rb') as gr_file:
            gr = pickle.load(gr_file)

        assert isinstance(gr, LCFRS)
        print("Grammar")
        print("Nonterminals:", len(gr.nonts()))
        print("Rules:", len(gr.rules()))
        print("Lexical rules:", sum(1 for r in gr.rules() if r.rhs() == []))

        lexical_categories = set()
        for r in gr.rules():
            if r.rhs() == []:
                lexical_categories.add(r.lhs().nont()[:-2])

        last_sm_cycle = stage_dict['last_sm_cycle']
        la_path = changepath(stage_dict['latent_annotations'][str(last_sm_cycle)])
        with open(la_path, 'rb') as la_file:
            la = pickle.load(la_file)

        print(len(la), len(la[0]), len(la[1]), len(la[2]))

        terminal_labeling_path = changepath(stage_dict["terminal_labeling"])
        with open(terminal_labeling_path, "r") as tlf:
            terminal_labeling = deserialize_labeling(json.load(tlf))

        training_reducts = PySDCPTraceManager(gr, terminal_labeling)
        training_reducts_path = changepath(stage_dict["training_reducts"])
        training_reducts.load_traces_from_file(bytes(training_reducts_path, encoding="utf-8"))

        enumerator = training_reducts.get_nonterminal_map()

        if showpos:
            print("All lexical categories:")
            print(' '.join(sorted(lexical_categories)))

        if posinfo or allposinfo:
            print("\nLexical rule statistics:")
            for pos in sorted(lexical_categories):
                if allposinfo or pos in posinfo.split(','):
                    idx = enumerator.object_index(pos + '/1')
                    print(pos, "idx:", idx, "las", la[0][idx])
                    maximal_vals = [[0.0 for _ in range(number)] for _ in range(la[0][idx])]
                    maximal_lex_rules = [[None for _ in range(number)] for _ in range(la[0][idx])]
                    for r in gr.rules():
                        if r.rhs() == [] and r.lhs().nont() == pos + '/1':
                            # print(r, r.get_idx(), end=' ')
                            # print(la[2][r.get_idx()])
                            for v, prob in enumerate(la[2][r.get_idx()]):
                                if maximal_vals[v][-1] < prob:
                                    for i in range(number):
                                        if maximal_vals[v][i] < prob:
                                            maximal_vals[v] = maximal_vals[v][0:i] + [prob] + maximal_vals[v][i:-1]
                                            maximal_lex_rules[v] = maximal_lex_rules[v][0:i] + [r] + maximal_lex_rules[v][i:-1]
                                            break
                    for v, r, p in zip(range(la[0][idx]), maximal_lex_rules, maximal_vals):
                        print(v, end=' ')
                        for rr, pp in zip(r, p):
                            if rr is not None:
                                print('[', rr.lhs().arg(0)[0], ' – ', rr.dcp()[0].rhs()[0].head().edge_label(), ' # ', pp, ']', end=' ')
                        print()
                    print()

    else:
        print('No stage file found in', rootpath)


plac.call(main)

