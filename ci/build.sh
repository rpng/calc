#!/bin/bash

cd DeepLCD && mkdir build && cd build
# Build caffe in our build directory
git clone https://github.com/nmerrill67/caffe.git # stable caffe fork

# configure for CPU_ONLY
sed -i '/caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA/c\caffe_option(CPU_ONLY  "Build Caffe without CUDA support" ON) # TODO: rename to USE_CUDA' caffe/CMakeLists.txt
# CPU_ONLY somehow does not carry out to this build target from a caffe build (This is not just a travis issue), so we just redefine it here
sed -i '1s/^/#define CPU_ONLY 1\n/' caffe/include/caffe/util/device_alternate.hpp
cd caffe && mkdir build && cd build && cmake -DBLAS=open ..
make && sudo make install
# back to DeepLCD/build
cd ../..

# now build DeepLCD
cmake -DCaffe_ROOT_DIR=$PWD/caffe .. && make # This is an example of using Caffe_ROOT_DIR since caffe isn't in ~/caffe
