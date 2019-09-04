cd ..

rm -rf arches-kokkos-cuda
mkdir arches-kokkos-cuda

cd arches-kokkos-cuda
../src-char-ox/configure \
    --enable-64bit \
    --enable-optimize="-std=c++11 -g -O2" \
    --enable-assertion-level=0 \
    --with-kokkos=/usr/local/kokkos-2.7.00/opt-cuda/ \
    --with-cuda=/usr/local/cuda-9.1/ \
    --enable-gencode=52 \
    --enable-examples \
    --enable-arches \
    --with-boost=/usr/local/boost-1.65.1 \
    --with-hypre=/usr/local/hypre-2.8.0b \
    --without-petsc \
    --with-mpi=/usr/lib/mpich \
    CC=mpicc \
    CXX=/usr/local/kokkos-2.7.00/nvcc_wrapper \
    CXXFLAGS='-DKOKKOS_ENABLE_CUDA_LAMBDA --expt-extended-lambda -Wno-deprecated -Wno-unused-local-typedefs' \
    LDFLAGS='-ldl' \
    F77=gfortran
make -j32 sus
cd StandAlone
cp sus ../../tmp/sus.arches-kokkos-cuda

