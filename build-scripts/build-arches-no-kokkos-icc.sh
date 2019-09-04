cd ..

rm -rf arches-no-kokkos
mkdir arches-no-kokkos

cd arches-no-kokkos
../src-char-ox/configure \
    --enable-64bit \
    --enable-optimize="-std=c++11 -g -O2 -mt_mpi" \
    --enable-assertion-level=0 \
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
make -j32 sus compare_uda
cd StandAlone
cp sus ../../tmp/sus.arches-no-kokkos
cp compare_uda ../../tmp/compare_uda

