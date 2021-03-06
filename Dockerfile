FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Kiev
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "Timezone..." && \
    apt-get update && apt-get install -y tzdata

RUN echo "Installing dependencies..." && \
	apt-get -y update && \
	apt-get install -y build-essential \
    cmake \
    git \
    wget \
    vim \
    software-properties-common \
    libatlas-base-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    pciutils \
    python3-setuptools \
    python3-dev \
    python3-pip \
    opencl-headers \
    ocl-icd-opencl-dev \
    libviennacl-dev \
    libcanberra-gtk-module


# Update Cmake
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt update -y && \
    apt install cmake --upgrade -y

RUN apt-get install -y libopencv-dev python3-opencv

# Pip3
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install numpy scikit-build

# Get Bazel
RUN apt-get install curl gnupg -y && \
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
    mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    apt-get update -y && apt install bazel-4.2.1 -y && \
    ln -s /usr/bin/bazel-4.2.1 /usr/bin/bazel && \
    ldconfig

# Get Protocol buffers
RUN apt-get install autoconf automake libtool curl make g++ unzip -y && \
    mkdir /protocol_buffers && \
    cd /protocol_buffers && \
    wget -c https://github.com/protocolbuffers/protobuf/releases/download/v3.9.2/protobuf-all-3.9.2.tar.gz && \
    tar xvf protobuf-all-3.9.2.tar.gz && cd protobuf-3.9.2/ && \
    ./configure && \
    make -j8 && \
    make install && \
    ldconfig

ENV PYTHON_BIN_PATH=/usr/bin/python3
ENV PYTHON_LIB_PATH=/usr/lib/python3/dist-packages
ENV CUDA_TOOLKIT_PATH=/usr/local/cuda-11.2
ENV CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
ENV TF_NEED_GCP=0
ENV TF_NEED_CUDA=1
ENV TF_CUDA_VERSION=11.2
ENV TF_CUDA_COMPUTE_CAPABILITIES=7.5
ENV TF_NEED_HDFS=0
ENV TF_NEED_OPENCL=0
ENV TF_NEED_JEMALLOC=1
ENV TF_ENABLE_XLA=0
ENV TF_NEED_VERBS=0
ENV TF_CUDA_CLANG=0
ENV TF_CUDNN_VERSION=8.1
ENV TF_NEED_MKL=0
ENV TF_DOWNLOAD_MKL=0
ENV TF_NEED_AWS=0
ENV TF_NEED_MPI=0
ENV TF_NEED_GDR=0
ENV TF_NEED_S3=0
ENV TF_NEED_OPENCL_SYCL=0
ENV TF_SET_ANDROID_WORKSPACE=0
ENV TF_NEED_COMPUTECPP=0
ENV GCC_HOST_COMPILER_PATH=/usr/bin/gcc
ENV CC_OPT_FLAGS="-march=native"
ENV TF_NEED_KAFKA=0
ENV TF_NEED_TENSORRT=0

# Get and compile tensorflow
RUN git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout v2.8.0 && \
    ./configure

RUN apt-get install python-is-python3

RUN cd tensorflow && \
    bazel build --jobs=8 \
                --config=v2 \
                --copt=-O3 \
                --copt=-m64 \
                --copt=-march=native \
                --config=opt \
                --verbose_failures \
                //tensorflow:tensorflow_cc \
                //tensorflow:install_headers \
                //tensorflow:tensorflow \
                //tensorflow:tensorflow_framework \
                //tensorflow/tools/lib_package:libtensorflow

RUN  mkdir -p /opt/tensorflow/lib && \
     cp -r /tensorflow/bazel-bin/tensorflow/* /opt/tensorflow/lib/ && \
     cd /opt/tensorflow/lib && \
     ln -s libtensorflow_cc.so.2.8.0 libtensorflow_cc.so && \
     ln -s libtensorflow_cc.so.2.8.0 libtensorflow_cc.so.2 && \
     ln -s libtensorflow.so.2.8.0 libtensorflow.so && \
     ln -s libtensorflow.so.2.8.0 libtensorflow.so.2


ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/tensorflow/lib:$LD_LIBRARY_PATH

RUN ldconfig

# # Install OpenCV - only for running the example
RUN apt-get install build-essential -y && \
    apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y && \
    mkdir -p /opencv && \
    cd /opencv && \
    wget -c https://github.com/opencv/opencv/archive/4.3.0.zip && \
    unzip 4.3.0.zip && cd opencv-4.3.0 && \
    mkdir build && cd build &&\
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j`nproc`

RUN apt-get install -y neovim python3-neovim

RUN mkdir -p /root/.config/nvim/
COPY init.vim /root/.config/nvim/

RUN apt-get install fuse libfuse2 git python3-pip ack-grep -y
RUN curl -fLo /root/.local/share/nvim/site/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

RUN pip3 install --user neovim

RUN apt-get update -qq && apt-get -y install autoconf \
    automake build-essential cmake git-core libass-dev libfreetype6-dev \
    libgnutls28-dev libmp3lame-dev libsdl2-dev libtool libva-dev \
    libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
    meson ninja-build pkg-config texinfo wget yasm zlib1g-dev

RUN apt-get -y install libunistring-dev libaom-dev nasm libx264-dev libx265-dev \
    libnuma-dev libvpx-dev libfdk-aac-dev libopus-dev

RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg && \
    cd ffmpeg && git checkout origin/release/4.0 && \
    ./configure \
    --pkg-config-flags="--static" \
    --enable-shared \
    --extra-libs="-lpthread -lm" \
    --ld="g++" \
    --enable-gpl \
    --enable-gnutls \
    --enable-libaom \
    --enable-libass \
    --enable-libfdk-aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree && \
    make -j8 && make install && hash -r

WORKDIR root/
