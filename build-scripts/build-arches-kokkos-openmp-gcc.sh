cd ..

rm -rf arches-kokkos-openmp
mkdir arches-kokkos-openmp

cd arches-kokkos-openmp
../src-char-ox/configure \
    --enable-64bit \
    --enable-optimize="-std=c++11 -g -O2" \
    --enable-assertion-level=0 \
    --enable-kokkos \
    --enable-examples \
    --enable-arches \
    --with-boost=/usr/local/boost-1.65.1 \
    --with-hypre=/usr/local/hypre-2.8.0b \
    --without-petsc \
    --with-mpi=/usr/lib/mpich\
    CC=mpicc \
    CXX=mpic++ \
    CXXFLAGS="-Wno-deprecated -Wno-unused-local-typedefs" \
    LDFLAGS='-ldl' \
    F77=gfortran
make -j32 sus
cd StandAlone
cp sus ../../tmp/sus.arches-kokkos-openmp

