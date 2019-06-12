from __future__ import print_function
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from git import Repo
from Cython.Build import cythonize
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

dep_name = "sterm-parser"
cython_dependency_src_path = path.join(here, "build", dep_name)
sterm_include = [cython_dependency_src_path]

# change if eigen is installed in the user-local directory
# $COMPUTE_ROOT/usr/include/eigen3,
compute_root = ""
eigen_include = [compute_root + "/usr/include/eigen3", compute_root + "/usr/include"]
add_include = [compute_root + "/usr/local/include"]

# Schick Parser (mainly implemented by Timo Schick according to construction
# by Drewes, Gebhardt, & Vogler 2016)
schick_dep_name = "schick-parser"
schick_dependency_src_path = path.join(here, "build", schick_dep_name)
schick_executable = 'HypergraphReduct-1.0-SNAPSHOT.jar'


class CustomBuildExtCommand(build_ext):
    """Customized setuptools install command, builds Timo Schick's parser."""
    def run(self):
        # self.build_schick_parser()
        self.download_schick_parser()
        build_ext.run(self)

    def download_schick_parser(self):
        if not path.exists("util/HypergraphReduct-1.0-SNAPSHOT.jar"):
            x = subprocess.call(["wget", "https://wwwtcs.inf.tu-dresden.de/~kilian/HypergraphReduct-1.0-SNAPSHOT.jar", "-P", "util"])
            assert x == 0

    def build_schick_parser(self):
        print("Building " + schick_dep_name + " using Maven.")

        p = subprocess.Popen(['mvn clean package'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=schick_dependency_src_path)
        for line in p.stdout.readlines():
            print(line,)
        retval = p.wait()
        if retval == 0:
            p = subprocess.Popen(' '.join(['cp', path.join(schick_dependency_src_path, 'target', schick_executable), path.join(here, 'util')]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            for line in p.stdout.readlines():
                print(line,)
            retval = p.wait()
        return retval


extra_compile_args = ["-std=c++17", "-gdwarf-3", "-Wall", "-rdynamic"]
# openmp = ["-fopenmp", "-lpthread"]
openmp = []
# optimizations = ["-O3"]
optimizations = []
# optimizations_tensors = ["-O3", "-fdump-tree-optimized", "-ftree-vectorizer-verbose=2", "-ftree-vectorize", "-march=native"]
optimizations_tensors = ["-mfpmath=sse", "-msse2"]
linker_args = ["-rdynamic"]

ext_modules = [
    Extension("grammar.induction.decomposition", ["grammar/induction/decomposition.pyx"], language='c++'),
    Extension("parser.fst.lazy_composition", ["parser/fst/lazy_composition.pyx"], language='c++',
              extra_compile_args=['-std=c++14', '-lfst', '-ldl'], extra_link_args=['-lfst', '-ldl'],
              include_dirs=add_include),
    Extension("util.enumerator", sources=["util/enumerator.pyx"], language='c++'),
    Extension("grammar.lcfrs",  ["grammar/lcfrs.pyx"]),
    Extension("parser.cpp_cfg_parser.parser_wrapper",
              sources=["parser/cpp_cfg_parser/parser_wrapper.pyx", "parser/cpp_cfg_parser/cfg.cpp",
                       "parser/cpp_cfg_parser/parser.cpp"],
              language='c++', extra_compile_args=extra_compile_args, extra_link_args=linker_args),
    Extension("parser.sDCP_parser.sdcp_parser_wrapper", sources=["parser/sDCP_parser/sdcp_parser_wrapper.pyx"], language='c++', extra_compile_args=extra_compile_args + optimizations, extra_link_args=linker_args, include_dirs=sterm_include
              ),
    Extension("parser.commons.commons", sources=["parser/commons/commons.pyx"], language='c++',
              extra_compile_args=extra_compile_args + optimizations, extra_link_args=linker_args,
              include_dirs=sterm_include),
    Extension("parser.LCFRS.LCFRS_Parser_Wrapper", sources=["parser/LCFRS/LCFRS_Parser_Wrapper.pyx"], language='c++',
              extra_compile_args=extra_compile_args + optimizations, extra_link_args=linker_args,
              include_dirs=sterm_include),
    Extension("parser.trace_manager.trace_manager", sources=["parser/trace_manager/trace_manager.pyx"], language='c++',
              extra_compile_args= extra_compile_args + optimizations_tensors, extra_link_args=linker_args,
              include_dirs=eigen_include+sterm_include
              , undef_macros=["NDEBUG"]),
    Extension("parser.supervised_trainer.trainer", sources=["parser/supervised_trainer/trainer.pyx"], language='c++',
              extra_compile_args=extra_compile_args + optimizations, extra_link_args=linker_args,
              include_dirs=eigen_include+sterm_include
              , undef_macros=["NDEBUG"]),
    Extension("parser.LCFRS.LCFRS_trace_manager", sources=["parser/LCFRS/LCFRS_trace_manager.pyx"], language='c++',
              extra_compile_args=extra_compile_args + optimizations, include_dirs=eigen_include+sterm_include,
              extra_link_args=linker_args),
    Extension("parser.sDCP_parser.sdcp_trace_manager", sources=["parser/sDCP_parser/sdcp_trace_manager.pyx"], language='c++', extra_compile_args=extra_compile_args + optimizations, include_dirs=eigen_include+sterm_include),
    Extension("parser.trace_manager.sm_trainer_util", sources=["parser/trace_manager/sm_trainer_util.pyx"], language='c++', extra_compile_args=extra_compile_args + optimizations_tensors + openmp, extra_link_args=linker_args + openmp, include_dirs=eigen_include+sterm_include),
    Extension("parser.coarse_to_fine_parser.ranker", sources=["parser/coarse_to_fine_parser/ranker.pyx"], language='c++',
              extra_compile_args=extra_compile_args + openmp + optimizations_tensors, extra_link_args=linker_args + openmp, include_dirs=eigen_include+sterm_include),
    Extension("parser.trace_manager.score_validator", sources=["parser/trace_manager/score_validator.pyx"],
              language='c++', extra_compile_args=extra_compile_args + openmp + optimizations_tensors,
              extra_link_args=linker_args + openmp, include_dirs=eigen_include+sterm_include),
    Extension("parser.trace_manager.sm_trainer", sources=["parser/trace_manager/sm_trainer.pyx"], language='c++',
              extra_compile_args=extra_compile_args + openmp + optimizations_tensors,
              extra_link_args=linker_args + openmp, include_dirs=eigen_include+sterm_include
              , undef_macros=["NDEBUG"]),
    Extension("parser.lcfrs_la", sources=["parser/lcfrs_la.pyx"], language='c++',
              extra_compile_args=extra_compile_args + openmp + optimizations_tensors,
              extra_link_args=linker_args + openmp, include_dirs=eigen_include+sterm_include),
    Extension("parser.coarse_to_fine_parser.trace_weight_projection",
              sources=["parser/coarse_to_fine_parser/trace_weight_projection.pyx"],
              language='c++',
              extra_compile_args=extra_compile_args + optimizations_tensors,
              extra_link_args=linker_args, include_dirs=eigen_include+sterm_include
              , undef_macros=["NDEBUG"])
]

if __name__ == '__main__':
    setup(
        cmdclass={
            'build_ext': CustomBuildExtCommand,
        },
        name='panda-parser',
        version='0.3.0a0',
        description='Implementation of LCFRS/sDCP hybrid grammars',
        long_description=long_description,
        url='https://github.com/kilian-gebhardt/panda-parser',
        author='Kilian Gebhardt',
        author_email='kilian.gebhardt@tu-dresden.de',

        license='GNU General Public License (GPL)',

        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: NLP Researchers',
            'Topic :: Syntactic Parsing :: Grammar-based parsing formalisms',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: GNU General Public License (GPL)',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6'
        ],

        # What does your project relate to?
        keywords='parsing parse LCFRS hybrid grammar',

        packages=[],
        # install_requires=[],

        ext_modules=cythonize(ext_modules),
        requires=['Cython']
    )
