cd ..

rm -rf arches-kokkos-openmp
mkdir arches-kokkos-openmp

cd arches-kokkos-openmp
../src-char-ox/configure \
    --enable-64bit \
    --enable-optimize="-std=c++11 -g -O2 -mt_mpi" \
    --enable-assertion-level=0 \
    --enable-kokkos \
    --enable-examples \
    --enable-arches \
    --with-boost=/usr/local/ \
    --with-hypre=$HYPRE_PATH \
    --without-petsc \
    --with-mpi-include=/opt/intel/impi/2018.1.163/include64 \
    --with-mpi-lib=/opt/intel/impi/2018.1.163/lib64 \
    CC=mpiicc \
    CXX=mpiicpc \
    CXXFLAGS="-Wno-deprecated -Wno-unused-local-typedefs" \
    LDFLAGS='-ldl' \
    F77=mpiifort
make -j32 sus
cd StandAlone
cp sus ../../tmp/sus.arches-kokkos-openmp

