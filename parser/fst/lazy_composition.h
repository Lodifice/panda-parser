#include <fst/fstlib.h>
#include <vector>

using namespace fst;

StdVectorFst construct_fa(std::vector<std::string> sequence, const StdFst & fst) {
    StdVectorFst fsa;
    fsa.SetInputSymbols(fst.InputSymbols());
    fsa.SetOutputSymbols(fst.InputSymbols());
    auto state = fsa.AddState();
    fsa.SetStart(state);

    for(auto x : sequence) {
        auto nextState = fsa.AddState();
        auto code = fst.InputSymbols()->Find(x);
        fsa.AddArc(state, StdArc(code, code, 0, nextState));
        state = nextState;
    }
    fsa.SetFinal(state, 0);
    return fsa;
}

StdFst* readFst(std::string path) {
    return StdFst::Read(path);
}


void lazy_compose(const StdVectorFst & fst_a, const StdFst & fst_b) {
    auto lazy_composition = ComposeFst<StdArc>(fst_a, fst_b);
    StdVectorFst shortest_path;
    ShortestPath(lazy_composition, &shortest_path);
    shortest_path.Write("/tmp/shortest_path.fst");
}

std::vector<unsigned> lazy_compose_(const StdVectorFst & fst_a, const StdFst & fst_b) {
    auto lazy_composition = ComposeFst<StdArc>(fst_a, fst_b);
    StdVectorFst shortest_path;
    ShortestPath(lazy_composition, &shortest_path);
    TopSort(&shortest_path);

    std::vector<unsigned> rules;
    for (StateIterator<StdVectorFst> siter(shortest_path); !siter.Done(); siter.Next()) {
        StdArc::StateId state = siter.Value();
        for (ArcIterator<StdVectorFst> aiter(shortest_path, state); !aiter.Done(); aiter.Next()) {
            const StdArc &arc = aiter.Value();
            rules.push_back(arc.olabel);
        }
    }
    return rules;
}
