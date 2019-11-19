FROM ubuntu:bionic

# Change versions of dependencies here
ARG EIGEN_REVISION=9e6bc1d
ARG OPENFST_VERSION=1.7.5
ARG PYNINI_VERSION=2.0.9
ARG GF_VERSION=3.10

# @Python: read files in UTF-8 encoding!
ENV LANG=C.UTF-8

# Install build environment and most dependencies via apt
ENV DEBIAN_FRONTEND noninteractive
RUN apt update \
    && apt -y install autoconf build-essential cmake git graphviz libncurses5-dev libre2-dev libtool mercurial python-dev python3-dev python3-numpy python3-pip wget zlib1g-dev \
    && apt -y autoremove \
    && apt -y clean \
    && rm -rf /var/lib/apt/lists/*

# Install eigen from source
RUN hg clone https://bitbucket.org/eigen/eigen/ -r "$EIGEN_REVISION"
RUN mkdir eigen/build
WORKDIR eigen/build
RUN cmake ..
RUN make install
WORKDIR /

# Install disco-dop from source
RUN git clone --recursive --branch=chart-exposure https://github.com/kilian-gebhardt/disco-dop
WORKDIR disco-dop
RUN pip3 install -r requirements.txt
RUN sed -i 's/install --user/install/' Makefile
RUN make install
WORKDIR /

# Install OpenFST from source
RUN wget "http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$OPENFST_VERSION.tar.gz"
RUN tar -xf "openfst-$OPENFST_VERSION.tar.gz"
WORKDIR "openfst-$OPENFST_VERSION"
RUN ./configure --enable-bin --enable-compact-fsts --enable-compress --enable-const-fsts --enable-far --enable-linear-fsts --enable-lookahead-fsts --enable-mpdt --enable-ngram-fsts --enable-pdt --enable-python PYTHON=python3
RUN make
RUN make install
WORKDIR /

# Install Pynini from source
RUN wget "http://www.openfst.org/twiki/pub/GRM/PyniniDownload/pynini-$PYNINI_VERSION.tar.gz"
RUN tar -xf "pynini-$PYNINI_VERSION.tar.gz"
WORKDIR "pynini-$PYNINI_VERSION"
RUN python3 setup.py install
WORKDIR /

# Install Grammatical Framework custom package
RUN wget "https://github.com/GrammaticalFramework/gf-core/archive/GF-$GF_VERSION.tar.gz"
RUN tar -xf "GF-$GF_VERSION.tar.gz"
WORKDIR "gf-core-GF-$GF_VERSION/src/runtime/c"
RUN bash setup.sh configure
RUN bash setup.sh build
RUN bash setup.sh install
WORKDIR "/gf-core-GF-$GF_VERSION/src/runtime/python"
RUN python3 setup.py build
RUN python3 setup.py install
WORKDIR /

# Install Boost from source
ARG BOOST_VERSION=1.69.0
RUN wget "https://dl.bintray.com/boostorg/release/$BOOST_VERSION/source/boost_$(echo $BOOST_VERSION | sed 's/\./_/g').tar.gz"
RUN tar -xf "boost_$(echo $BOOST_VERSION | sed 's/\./_/g').tar.gz"
RUN cd "boost_$(echo $BOOST_VERSION | sed 's/\./_/g')" && ./bootstrap.sh --with-libraries= && ./b2 install || true

# Build Panda parser
ADD . /panda-parser
WORKDIR panda-parser
RUN pip3 install -r requirements.txt
RUN python3 setup.py build_ext --inplace
