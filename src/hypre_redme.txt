This contains some build instruction notes for setting up and running more efficient hypre on this Kokkos dev branch.

hypre_cuda_changes_v2.patch in this directory merges cuda kernels in Hypre and gives upto 30% speeup. Use this ONLY for hypre cuda.

Installation insturctions:

mkdir hypre
cd hypre
git clone https://github.com/hypre-space/hypre.git hypre_cuda
cd hypre_cuda
#works for v2.15.1 only
git checkout v2.15.1
git apply ../hypre_cuda_changes_v2.patch
cd src
mkdir build
./configure --prefix=`pwd`/build CC=mpicc CXX=mpic++ F77=mpif77 CFLAGS="-fPIC -O2 -g " CXXFLAGS="-fPIC -O2 -g " CUFLAGS="-lineinfo " --enable-shared --with-cuda HYPRE_CUDA_SM=70
make -j32 install

