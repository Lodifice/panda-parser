from libcpp.vector cimport vector
from libcpp.memory cimport make_shared, shared_ptr
from parser.trace_manager.trace_manager cimport PyTraceManager, TraceManagerPtr, build_trace_manager_ptr, Element, Hypergraph, Node, HyperEdge
from util.enumerator cimport Enumerator

ctypedef size_t NONTERMINAL

cdef class PyDerivationManager(PyTraceManager):
    cdef shared_ptr[vector[NONTERMINAL]] node_labels
    cdef shared_ptr[vector[size_t]] edge_labels
    cdef Enumerator nonterminal_map

    cpdef void convert_derivations_to_hypergraphs(self, corpus, float frequency=?)
    cpdef void convert_derivations_to_hypergraph(self, corpus, float frequency=?)
    cpdef void convert_rtgs_to_hypergraphs(self, rtgs, float frequency=?)
    cpdef void convert_chart_to_hypergraph(self, chart, disco_grammar, bint debug=?) except +

    cpdef Enumerator get_nonterminal_map(self)