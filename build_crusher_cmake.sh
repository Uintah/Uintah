#!/bin/bash

module load cmake PrgEnv-amd boost
module -t list

rm -rf build
mkdir build

cd build

cmake -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DENABLE_HIP=ON -DENABLE_EXAMPLES=ON ../src/

cmake --build . -j 32

cd StandAlone

cp sus ../../tmp/sus
cp compare_uda ../../tmp/compare_uda
