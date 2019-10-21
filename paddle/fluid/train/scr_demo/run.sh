#!/bin/bash

set -x

PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas

# build demo trainer
fluid_install_dir=${PADDLE_ROOT}/build/fluid_install_dir

mkdir -p build
cd build
rm -rf *
cmake .. -DPADDLE_LIB=$fluid_install_dir \
         -DWITH_MKLDNN=$TURN_ON_MKL \
         -DWITH_MKL=$TURN_ON_MKL \
	 -DCMAKE_C_COMPILER=/usr/bin/gcc-4.8 \
	 -DCMAKE_CXX_COMPILER=/usr/bin/g++-4.8 \
make

cd ..

# run demo trainer
build/resnet50_trainer --train_data=train.bin --batch_size=1 --iterations=50 --epochs=1
