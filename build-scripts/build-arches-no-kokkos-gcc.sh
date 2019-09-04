cd ..

rm -rf arches-no-kokkos
mkdir arches-no-kokkos

cd arches-no-kokkos
../src-char-ox/configure \
    --enable-64bit \
    --enable-optimize="-std=c++11 -g -O2" \
    --enable-assertion-level=0 \
    --enable-examples \
    --enable-arches \
    --with-boost=/usr/local/boost-1.65.1 \
    --with-hypre=/usr/local/hypre-2.8.0b \
    --without-petsc \
    --with-mpi=/usr/lib/mpich \
    CC=mpicc \
    CXX=mpic++ \
    CXXFLAGS="-Wno-deprecated -Wno-unused-local-typedefs" \
    LDFLAGS='-ldl' \
    F77=gfortran
make -j32 sus compare_uda
cd StandAlone
cp sus ../../tmp/sus.arches-no-kokkos
cp compare_uda ../../tmp/compare_uda

