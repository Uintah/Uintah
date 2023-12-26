This README contains Brad's build instruction notes for getting up and
running on this Kokkos dev branch. Somewhat old but still relavent.

Clone Kokkos (I put mine in my ~/src directory):

git clone https://github.com/kokkos/kokkos.git ~/src/kokkos

Make a Kokkos folder (I put mine in ~/opt):

mkdir ~/opt/kokkos-openmp
mkdir ~/opt/kokkos-openmp/build
cd ~/opt/kokkos-openmp/build

-----------------------------------------------------------------------------------------------------------------------------
Here is how I configure Kokkos for OpenMP

~/src/kokkos/generate_makefile.bash --kokkos-path=/home/brad/src/kokkos --prefix=/home/brad/opt/kokkos-openmp --with-openmp

make install
-----------------------------------------------------------------------------------------------------------------------------
Here is how I configure Kokkos for both OpenMP and CUDA

~/src/kokkos/generate_makefile.bash --kokkos-path=/home/brad/src/kokkos --prefix=/home/brad/opt/kokkos-openmp-cuda --compiler=/home/brad/src/kokkos/bin/nvcc_wrapper --with-openmp --with-cuda=/home/brad/opt/cuda-9.0 --arch=Maxwell50

make install
-----------------------------------------------------------------------------------------------------------------------------

Then in Uintah, just add a --with-kokkos line to your Uintah configure pointing to the Kokkos build:  

--with-kokkos=/home/brad/opt/kokkos-openmp

This is also needed in the Uintah configure: 

LDFLAGS='-ldl'

For GPU builds, ensure the CXX is at the nvcc_wrapper path:

CXX=/home/brad/opt/kokkos-openmp-cuda/bin/nvcc_wrapper 

Also, GPU builds need to support lambda expressions:

CXXFLAGS='-DKOKKOS_ENABLE_CUDA_LAMBDA --expt-extended-lambda'

 
Below is what I have in my Uintah configure for my GPU build:

    C=gcc \
    CXX=$homedir/opt/kokkos-openmp-cuda/bin/nvcc_wrapper \
    CXXFLAGS='-DKOKKOS_ENABLE_CUDA_LAMBDA --expt-extended-lambda' \
    LDFLAGS='-ldl'

-----------------------------------------------------------------------------------------------------------------------------
When building with Kokkos the KokkosOpenMP scheduler is the
default. Both the MPI scheduler and the Unified Scheduler work but
require the -cpu flag

MPI scheduler:                sus -cpu poisson1.ups
Unified scheduler:            sus -cpu -nthreads 1 poisson1.ups
Kokkos Scheduler with GPU:    sus -gpu poission1.ups
KokkosOpenMP Scheduler:       sus poission1.ups

-----------------------------------------------------------------------------------------------------------------------------
Side note: for an example of building a single Kokkos test program with both CUDA and OpenMP support:

/home/brad/opt/kokkos-openmp-cuda/bin/nvcc_wrapper -DKOKKOS_ENABLE_CUDA_LAMBDA --expt-extended-lambda -I./ -I/home/brad/opt/kokkos-openmp-cuda/include -I/opt/cuda-9.0/include -L/home/brad/opt/kokkos-openmp-cuda/lib -L/opt/cuda-9.0/lib64 -fopenmp  -lkokkos -ldl -lcudart -lcuda --std=c++11 -arch=sm_50 -Xcompiler -fopenmp -O3 some_code_file.cc -o some_code_file.x

-----------------------------------------------------------------------------------------------------------------------------
