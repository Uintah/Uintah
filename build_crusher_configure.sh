#!/bin/bash

module load rocm/5.2.0 cray-mpich/8.1.23 craype/2.7.19 kokkos/3.6.00
#module load PrgEnv-amd cray-mpich/8.1.23 craype/2.7.19 kokkos/3.6.00
#module load PrgEnv-cray rocm/5.4.0 cray-mpich/8.1.23 craype/2.7.19 kokkos/3.6.00
module -t list

rm -rf build
mkdir build

cd build

../src/configure \
  --enable-64bit \
  --enable-optimize="-g -Og" \
  --enable-examples \
  --with-kokkos=$OLCF_KOKKOS_ROOT \
  --with-mpi=$MPICH_DIR \
  CC=hipcc \
  CXX=hipcc \
  CXXFLAGS='-std=c++17 -Wno-deprecated -Wno-unused-local-typedefs -DUSING_LATEST_KOKKOS -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -DHAVE_HIP -I/opt/rocm-5.2.0/include --rocm-path=/opt/rocm-5.2.0' \
  LDFLAGS='-ldl --rocm-path=/opt/rocm-5.2.0 -L/opt/rocm-5.2.0/lib -lamdhip64' \
  F77=ftn


#  CXXFLAGS='-std=c++17 -Wno-deprecated -Wno-unused-local-typedefs -DFIXED_RANDOM_NUM -DUSING_LATEST_KOKKOS -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 -D__HIP_PLATFORM_AMD__ -DHAVE_HIP -I/opt/rocm-5.4.0/include --rocm-path=/opt/rocm-5.4.0 -x hip' \
#	    LDFLAGS='-ldl --rocm-path=/opt/rocm-5.4.0 -L/opt/rocm-5.4.0/lib -lamdhip64' \

make -j8 sus compare_uda

cd StandAlone

cp sus ../../tmp/sus
cp compare_uda ../../tmp/compare_uda
