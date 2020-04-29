# INSTALL GUIDE

## Via Docker

Build an image, the current directory is added as the top layer.

`docker build --tag=panda-parser .`

Obtain a shell in the container.

`docker run --rm -it panda-parser bash`

## General prerequisites

`python3.5` or newer, `sqlite3`, `pip3`, `re2` (`libre2-dev` on ubuntu)

------------------------------------------------------------------------

## Install python dependencies from PyPI
1. `cd` project dir
2. `pip3 install -r requirements.txt`

------------------------------------------------------------------------

## Eigen
Install the package `eigen3-hg` (AUR) or install from [source](http://eigen.tuxfamily.org).

If installation in non-standard location is required, set root of local installation to a user-owned directory `COMPUTE_ROOT="/compute/user"`. Otherwise, the `-CMAKE_INSTALL_PREFIX=$COMPUTE_ROOT/usr` option to cmake is not necessary.

```
hg clone https://bitbucket.org/eigen/eigen/ -r 9e6bc1d
cd eigen
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$COMPUTE_ROOT/usr
make install # sudo permissions might be necessary on Linux.
cd ../..
```

------------------------------------------------------------------------

## Install disco-dop with exposed charts
 
Install the patched version of `disco-dop` from branch [chart-exposure](https://github.com/kilian-gebhardt/disco-dop/tree/chart-exposure).

------------------------------------------------------------------------

## Compile cython files
This requires Eigen to be installed (see above):

1. `cd` project dir
2. modify `setup.py`, if eigen is installed in non-standard location. This can be done by setting the `compute_root` variable.
2. `python3 setup.py build_ext --inplace`


------------------------------------------------------------------------

## 3rd party parsing back-ends

### configure environment variables
We describe a user-local installation without root privileges. You want to add the following to your `~/.bashrc` or a similar place.
0. `COMPUTE_ROOT="/compute/user"` # Set root of local installation to a user-owned directory
1. `export LD_LIBRARY_PATH="$COMPUTE_ROOT/usr/local/lib:$COMPUTE_ROOT/usr/local/include:$LD_LIBRARY_PATH"`
2. `export LIBRARY_PATH="$COMPUTE_ROOT/usr/local/lib:$COMPUTE_ROOT/usr/local/include:$LIBRARY_PATH"`
2. `export PATH="$COMPUTE_ROOT/usr/local/bin:$PATH"`
2.  `export C_INCLUDE_PATH="$COMPUTE_ROOT/usr/local/include:$C_INCLUDE_PATH"` #probably not necessary

### OpenFst with pynini
Hybridgrammars induced with the left-branching or right-branching recursive partitioning can be represented as a finite state transducer.
Hence, the OpenFst framework can be used for efficient parsing.

#### Install OpenFST
First download OpenFST: http://www.openfst.org/twiki/bin/view/FST/FstDownload

0. `COMPUTE_ROOT="/compute/user"` # Set root of local installation to a user-owned directory
1. `cd openfst`
2. `./configure --prefix=$COMPUTE_ROOT/usr/local --enable-bin --enable-compact-fsts --enable-compress --enable-const-fsts --enable-far --enable-linear-fsts --enable-lookahead-fsts --enable-mpdt --enable-ngram-fsts --enable-pdt --enable-python PYTHON=python3`
3. `make` # this takes some time
4. `make install`

#### Install pynini
Download pynini: http://www.openfst.org/twiki/bin/view/GRM/PyniniDownload

1. `cd pynini`
2.  edit `setup.py`: add `"-I/compute/user/usr/local/include"` to COMPILE_ARGS (adapt path to COMPUTE_ROOT)
3. `python3 setup.py install --user` # This may take a while

------------------------------------------------------------------------

### Grammatical Framework (GF)
The Grammatical Framework ships with a powerful LCFRS parsing backend, which can be facilitated for hybrid grammars.

For Debian there is a package which also includes the C-runtime and python bindings, cf. http://www.grammaticalframework.org/download/ 

Alternatively you can install GF via cabal: 
`cabal install gf`The cabal installation has the following ubuntu dependencies: `libghc-terminfo-dev`, `happy`, `alex`.
Make sure to add `~/.cabal/bin` to the PATH variable.

### GF C-runtime with python bindings
If not already installed via the debian package, one can do a user-local installation of the C-runtime and python bindings.

#### obtain sources 
- stable sources on http://www.grammaticalframework.org/download/ 
- tested with http://www.grammaticalframework.org/download/gf-3.8.tar.gz
- unpack archive `tar -xf gf-3.8.tar.gz`

#### compile and install C-runtime
0. `COMPUTE_ROOT="/compute/user"` # Set root of local installation to a user-owned directory
1.  `cd gf/src/runtime/c/`
2. `autoconf -i` (If this results in `error: possibly undefined macro` errors, run `autoreconf --install` and repeat.)
1.  `bash setup.sh configure`
1.  `bash setup.sh build`
3. `./configure --prefix=$COMPUTE_ROOT/usr/local`
4. `make clean`
5. `make`
5. `make install`

I haven't figured out the most efficient sequence yet. Leaving out the `bash setup.sh ...` steps, makes `./configure --prefix=...`fail. Perhaps only the configure step is necessary.

#### compile python binding
1. `COMPUTE_ROOT="/compute/user"` # Set root of local installation to a user-owned directory
2. `export EXTRA_INCLUDE_DIRS="$COMPUTE_ROOT/usr/local/include"`
3. `export EXTRA_LIB_DIRS="$COMPUTE_ROOT/usr/local/lib"`
4. `cd gf/src/runtime/python/`
5. `python setup.py build`
6. `python setup.py install --user`

I did not find the two exports to be necessary on every system though. They don't need to be permanent.

#### check if installation was successful:
1. `python`
2. `import pgf`
