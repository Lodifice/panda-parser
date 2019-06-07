import os, sys, plac, json, pickle
from grammar.lcfrs import *
from parser.sDCP_parser.sdcp_trace_manager import PySDCPTraceManager
from grammar.induction.terminal_labeling import deserialize_labeling


@plac.annotations(
    directory=('directory in which experiment is run (default: .)', 'option', 'd', str),
    posinfo=('comma-separated list of POS to show statistics on (default: "")', 'option', None, str)
)
def main(directory: str = '.', posinfo: str = ''):
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

        print("All lexical categories:")
        print(' '.join(sorted(lexical_categories)))

        print("\nLexical rule statistics:")
        for pos in sorted(lexical_categories):
            if pos in posinfo.split(','):
                idx = enumerator.object_index(pos + '/1')
                print(pos, "idx:", idx, "las", la[0][idx])
                maximal_vals = [0.0 for _ in range(la[0][idx])]
                maximal_lex_rules = [None for _ in range(la[0][idx])]
                for r in gr.rules():
                    if r.rhs() == [] and r.lhs().nont() == pos + '/1':
                        # print(r, r.get_idx(), end=' ')
                        # print(la[2][r.get_idx()])
                        for v, prob in enumerate(la[2][r.get_idx()]):
                            if maximal_vals[v] < prob:
                                maximal_vals[v] = prob
                                maximal_lex_rules[v] = r
                for v, r, p in zip(range(la[0][idx]), maximal_lex_rules, maximal_vals):
                    print(v, r.lhs().arg(0)[0], p)
                print()

    else:
        print('No stage file found in', rootpath)


plac.call(main)

