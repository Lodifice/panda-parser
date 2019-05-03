panda-parser
============================

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 2 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

### List of contributors

-   Kilian Gebhardt
-   Markus Teichmann
-   Johann Seltmann
-   Sebastian Mielke
-   Kevin Mitlöhner
-   Mark-Jan Nederhof (this is an extension of his [software](http://mjn.host.cs.st-andrews.ac.uk/code/coling2014.zip))


------------------------------------------------------------------------

## Overview

Panda parser was developed in the context of research
on *hybrid grammars* and grammar-based approaches to 
natural language processing in general. It is a 
**p**arsing **a**rchitecture (especially) for **n**on-projective and **d**iscontinuous structures
powered by universal **a**lgebra. Moreover, pandas are said to live in (parse) trees ;-).

Currently implemented grammar types:
- hybrid grammars coupling *linear context-free rewriting systems* (LCFRS) 
  and *(simple) definite clause programs* ((s)DCP)
- plain LCFRS
- aligned LCFRS/graph grammars (aligned hypergraph bimorphism)
- subclasses of LCFRS, such as *context-free grammars* (CFG) and *finite state automata* (FSA)

Functionality:
  - grammar induction from a corpus
  - parsing of sentences to trees / graphs
  - computation of reducts (intersection of a sentence/tree pair with a grammar)
  - expectation maximization (EM) training
  - automatic grammar refinement by the split/merge algorithm

Some of these concepts/algorithms have been described in the following articles:
- LCFRS/sDCP hybrid grammars and induction from discontinuous constituent structures 
    [[Nederhof/Vogler 2014]](https://www.aclweb.org/anthology/C14-1130)
- aligned hypergraph bimorphism, reducts and EM training [[Drewes/Gebhardt/Vogler 2016]](https://www.aclweb.org/anthology/W16-2407)
- general hybrid grammars, induction of LCFRS/sDCP hybrid grammars from discontinuous constituent trees and
    non-projective dependency trees [[Gebhardt/Nederhof/Vogler 2017]](https://doi.org/10.1162/COLI_a_00291)
- generic split/merge training, in partiuclar for LCFRS and LCFRS/sDCP hybrid grammars 
    [[Gebhardt 2018 (preprint)]](https://wwwtcs.inf.tu-dresden.de/~kilian/coling2018preprint.pdf)
    
Due to the nature of the software development process (research), it was run on just 1-3 machines and should be considered unstable. Interfaces are likely to be changed as needed for future extensions. Maintenance 
is limited to Kilian's professional involvement in academic research.

------------------------------------------------------------------------

## Installation/ Preparation:

See [INSTALL.md](INSTALL.md).

------------------------------------------------------------------------

Resources / Corpora
-------------------

For many scripts it is assumed that certain corpora are available below the res directory. You
may want to download or symlink them there.

- `res/dependency_conll` -\> various corpora of [CONLL-X shared task](https://catalog.ldc.upenn.edu/LDC2015T11)

- `res/tiger/tiger_release_aug07.corrected.16012013.xml` -\> the [TiGer
corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.html)

- `res/negra-corpus/downloadv2/negra-corpus.{cfg,export}` -\> the
[Negra corpus](http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/negra-corpus.html)

- `res/negra-dep/negra-lower-punct-{train,test}.conll` -\> a conversion of Negra
in CONLL dependency format as described in [[Maier/Kallmeyer 2010]](http://www.wolfgang-maier.net/pub/tagplus10.pdf). The conversion requires [rparse](https://github.com/wmaier/rparse) and is automated in the script [util/prepare_negra_dep.py](util/prepare_negra_dep.py).

- `res/WSJ/ptb-discontinuous/dptb7[-km2003wsj].export` -\> discontinuous version of PTB/WSJ (contact [Kilian Evang](https://evang.ai/)). The file `dptb7-km2003wsj.export` is obtained by running `discodop treetransforms --transforms=km2003wsj dptb7.export dptb-km2003wsj.export`.

- `res/wsj_dependency/{02-22,23,24}.conll` -\> various sections of PTB/WSJ converted to CONLL dependency format

- `res/SPMRL_SHARED_2014_NO_ARABIC` -\> corpora from the [SPMRL 2014 shared task](http://www.spmrl.org/spmrl2014-sharedtask.html)

- `res/TIGER/tiger21` -\> Hall & Nivre 2008 (HN08) split of TiGer. Obtain TiGer version 2.1. Then run:
    ```
    mkdir tiger21
    unzip tigercorpus2.1.zip -d tiger21
    
    python3 tigersplit.py
    
    for corpus in  tigerdev tigertest tigertraindev tigertraintest
    do
        treetools transform tiger21/${corpus}.export tiger21/${corpus}_root_attach.export --trans root_attach
    done
    ``` 
    This is an excerpt of [Maximin Coavoux's script](https://github.com/mcoavoux/mtg/blob/master/mind_the_gap_v1.0/data/generate_tiger_data.sh) and uses his [tigersplit.py](https://github.com/mcoavoux/mtg/blob/master/mind_the_gap_v1.0/data/tigersplit.py).

- `res/TIGER/tigerHN08-dev.train.pred_tags.raw` and `tigerHN08-test.train+dev.pred_tags.raw`. Predicted POS-tags for TiGer HN08 split. These files are obtained by training mate-tools and using it to predict POS tags. Further details can be found in [util/prepare_pred_tags.sh](util/prepare_pred_tags.sh) (not fully automated yet). 


------------------------------------------------------------------------

## Unit tests

Running unit tests (requires some corpora to be available in the res
directory)

    python3 -m unittest discover

------------------------------------------------------------------------


## Documentation for the experiments in [[Gebhardt 2018 (preprint)]](https://wwwtcs.inf.tu-dresden.de/~kilian/coling2018preprint.pdf)

1. Obtain the data and locate it under `res/` as described above for SPMRL_SHARED_2014_NO_ARABIC, the TiGer HN08 split, and  predicted POS tags for TiGer HN08 split.
 
2. To run an end-to-end experiment with split/merge refinement for LCFRS/sDCP hybrid grammars, call

    `PYTHONPATH=. python3 experiment/hg_constituent_experiment.py`

    with appropriate parameters (see `--help`). E.g., for a small experiment that takes about 10 minutes on a typical desktop machine:

    `PYTHONPATH=. python3 experiment/hg_constituent_experiment.py HN08 -sm-cycles 2 -parsing-limit -quick`
3. To run an end-to-end experiment with split/merge refinement for LCFRS, call:

    `PYTHONPATH=. python3 experiment/lcfrs_parsing_experiment.py`
    
    with appropriate parameters (see `--help`). E.g., for a small experiment that takes about 10 minutes on a typical desktop machine:
    
    `PYTHONPATH=. python3 experiment/lcfrs_parsing_experiment.py HN08 -quick -sm-cycles 2 -merge-percentage 70.0 -parsing-limit`
    
Beware: Due to the comparably large initial grammar sizes, the memory consumption in the 5th split/merge cycle can exceed 32GB, in particular if multi-threading is enabled. The same holds for parsing the TiGer test sets with unrestricted sentence length (peak consumption about ~40GB). Parsing sentences up to length 40 should be feasible with 8GB RAM.

## Documentation for the experiments in [[Gebhardt/Nederhof/Vogler 2017]](https://doi.org/10.1162/COLI_a_00291)

Actually an older version of this software was used to run the experiments in
this paper. Still, experiments can be reproduced as follows: 

- install the optional dependencies `grammatical framework` and `pynini`. 
  NB: It is possible to run the experiments without these requirements, however, there may be large differences in run times and memory footprint which make full corpus evaluations infeasible. Also, the most probable parse might be ambiguous, i.e., 
      other parser implementations may select different parses.
- run the experiments as described below 

### Documentation for dependency parsing

Acquire a corpus in CoNLL-X shared task format, cf.
<http://ilk.uvt.nl/conll/post_task_data.html>

To run an experiment you have to create a configuration file of the
following format. With each line (that is not a comment) a parameter is
set. If one parameter is set multiple times, then the last value is
used. Each parameter where no default value is indicated needs to be
set.

    #                            This is a comment
    Database:                    path/to/experiment-db
    Training Corpus:             path/to/training/corpus
    Test Corpus:                 path/to/test/corpus
    Nonterminal Labeling:        child-cpos+deprel
    Terminal Labeling:           pos
    Recursive Partitioning:      left-branching
    Training Limit:              1000                   # default: unlimited
    Test Limit:                  200                    # default: unlimited
    Test Length Limit:           25                     # default: unlimited
    # Pre/Post-processing options
    Default Root DEPREL:         ROOT                   # default: do not overwrite
    Ignore Punctuation:          NO    # YES or NO      # default: NO
    Default Disconnected DEPREL: PUNC                   # default: _ (underscore)

------------------------------------------------------------------------

Implemented nonterminal labeling strategies:

    [strict|child|stricttop|childtop|empty]-[pos|cpos|deprel|cpos+deprel|pos+deprel]

Implemented terminal labeling strategies:

    [pos|cpos|form]

Implemented recursive partitioning strategies:

    [left-branching|right-branching|direct-extraction|cfg|fanout-$K]

where `$K > 0`

**Warning**: Ignoring punctuation is experimental and may raise errors if a punctuation symbol governs some non-punctuation word. Also, correct CoNLL export is not
    guaranteed in this case.

------------------------------------------------------------------------

Then run

    PYTHONPATH=. python3 experiment/cl_dependency_experiments.py path/to/configuration/file

This command will start the experiment. The program outputs various statistics on the grammar to stdout. Also, it writes the results in the experiment database. **Beware**: Also scores are printed, but they are calculated according to our own non-standard implementation. You may want to use the evaluation script described below which uses the standard CoNLL-X implementation. Also, in case of very short parse times, the run-time can be dominated by the database latency.

#### Evaluation

In order to list the contents of the experiment database run

    PYTHONPATH=. python3 evaluation/cl_dependency_evaluation.py path/to/database list

In order to generate a LaTeX file containing a table with various
statistics and scores run

    PYTHONPATH=. python3 evaluation/cl_dependency_evaluation.py path/to/database plot --experiments=$SELECTION --outfile=path/to/table.tex [--max-length=$N]

where `$SELECTION` is a list of natural numbers separated by `,` or
`-`, where each natural number references a row in the table and
`n-m` expands to `n,n+1, ..., m-1,m` and `n` needs to be
smaller than `m`. The order in the this list specifies the order in
the generated table. With `--max-length` a limit on the sentence
length can be specified, i.e., scores and parsing times will only
reflect sentences up to this length.

#### Dependency parsing, cascade experiment

In order to run the cascade experiment in [[Nederhof/Gebhardt/Vogler 2017, Table 3, p. 509]](https://doi.org/10.1162/COLI_a_00291),
run the following:
 
    PYTHONPATH=. python3 experiment/cl_dependency_cascade.py
   
No separate evaluation is required.

#### Control experiments with rparse
Clone and compile [rparse with patched GF-export](https://github.com/kilian-gebhardt/rparse) and copy `rparse.jar` to `./util/rparser.jar`. Then run

    PYTHONPATH=. python3 playground_rparse/process_rparse_grammar.py
    
with appropriate parameters, e.g., for Markovization v = 1 and h = 3 run:

    PYTHONPATH=. python3 playground_rparse/process_rparse_grammar.py res/negra-dep/negra-lower-punct-train.conll res/negra-dep/negra-lower-punct-test.conll /tmp/negra-v1-h3 -vMarkov 1 -hMarkov 3
        

------------------------------------------------------------------------

### Documentation for Constituent Parsing

Acquire the Tiger and/or the Negra corpus.

In `corpora/tiger_parse.py`: Change the definitions of `TIGER_DIR` and
`TIGER`.

In `corpora/negra_parse.py`: Change the definition of `NEGRA_DIRECTORY`.

Uncomment the relevant lines of `experiment/cl_constituent_experiment.py` to select the desired
experiments.

Run: `PYTHONPATH=. python3 experiment/cl_constituent_experiments.py`


## Acknowledgements 
This project is an extension of work by Mark-Jan Nederhof available [here](http://mjn.host.cs.st-andrews.ac.uk/code/coling2014.zip).

For parsing and preprocessing this project depends on the following libraries/packages: 
- [Grammatical Framework](https://www.grammaticalframework.org/)
- [OpenFST](http://www.openfst.org) / [pynini](http://www.openfst.org/twiki/bin/view/GRM/Pynini)
- [disco-dop](https://github.com/andreasvc/disco-dop/) by Andreas van Cranenburgh
- [treetools](https://github.com/wmaier/treetools) by Wolfgang Maier

The [C++ backend](https://github.com/kilian-gebhardt/panda-parser-backend) of the project (S/M training) uses the [Boost](https://www.boost.org) and [Eigen](http://eigen.tuxfamily.org) libraries.

A highly experimental part of the software employs an implementation of a graph parser (based on the algorithm by [[Drewes/Gebhardt/Vogler 2016]](https://www.aclweb.org/anthology/W16-2407)). Currently we provide only a [Java Binary](https://wwwtcs.inf.tu-dresden.de/~kilian/HypergraphReduct-1.0-SNAPSHOT.jar) of this software that was mainly developed by [Timo Schick](https://github.com/timoschick).

Evaluation and utility scripts as well as parameter files are provided unter `./util`:
- `eval.pl`, `blanks2tab.py`, `conlltab2dot.py`, `tabs2blanks.py`, and `validateFormat.py` from the CoNLL-X shared task
- `proper.prm`, `negra.headrules`, and `ptb.headrules` taken from [disco-dop](https://github.com/andreasvc/disco-dop/) 
