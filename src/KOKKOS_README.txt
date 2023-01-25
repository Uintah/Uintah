This contains some build instruction notes for getting up and running on this Kokkos dev branch.

Grab Kokkos (I put mine in my ~/src directory):

git clone https://github.com/kokkos/kokkos.git ~/src/kokkos

Checkout Kokkos 2.7.00 if planning to use patches for asynchronous CUDA kernel execution:

git checkout 2.7.00

Make a Kokkos folder (I put mine in ~/opt):

mkdir ~/opt/kokkos-openmp
mkdir ~/opt/kokkos-openmp/build
cd ~/opt/kokkos-openmp/build

-----------------------------------------------------------------------------------------------------------------------------
Here is how I configure Kokkos for OpenMP

~/src/kokkos/generate_makefile.bash --kokkos-path=/home/brad/src/kokkos --prefix=/home/brad/opt/kokkos-openmp --with-openmp

make install
-----------------------------------------------------------------------------------------------------------------------------
Here is how I configure Kokkos for CUDA

And here is how I configured Kokkos for both OpenMP and CUDA

1) Apply my Kokkos patch for asynchronous, I have placed it in this Uintah branch's src directory. 

git apply /path/to/patch/kokkos_2.7.00_async.patch (works from anywhere inside the kokkos source tree)

Notes on available patches:
 - kokkos_2.7.00_async.patch, kokkos_brad_jun082018_deprecated.patch, and kokkos_brad_oct122018_deprecated.patch apply to Kokkos release 2.7.00.
 - Kokkos::parallel_reduce correctness issues were identified in kokkos_brad_jun082018_deprecated.patch.
 - Kokkos::parallel_reduce performance issues were identified in kokkos_brad_oct122018_deprecated.patch.
 - These are addressed in kokkos_2.7.00_async.patch, which is the preferred patch for kokkos_dev's latest use.

2) Double check Kokkos's Makefile script, its bugged: 

vim ~/src/kokkos/core/src/Makefile

Make sure the ? on line 11 is removed so it reads:
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
else
  CXX ?= g++
endif

Now the Kokkos configure can proceed:

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
When you build Uintah, make sure you specify sus: "make -j8 sus" and not just "make -j8", the Kokkos build fails on other Uintah things not related to building Kokkos with Uintah.

Both the MPI scheduler and the Unified Scheduler work.  As a reminder, -nthreads triggers the scheduler

MPI scheduler:                sus poisson1.ups
Unified scheduler:            sus -nthreads 1 poisson1.ups
Unified Scheduler with GPU:   sus -nthreads 16 -gpu poission1.ups
Mixture of CPU and GPU tasks: sus -nthreads 1 -gpu poisson1.ups      (triggers the Unified Scheduler)

The mixture requires only 1 thread due to a design flaw we're currently fixing.  OpenMP threads are bounded to a master thread, if more pthreads are used, some OpenMP loops will attempt to run on pthreads that aren't the master thread, and it crashes.  

-----------------------------------------------------------------------------------------------------------------------------
Side note: for an example of building a single Kokkos test program with both CUDA and OpenMP support:

/home/brad/opt/kokkos-openmp-cuda/bin/nvcc_wrapper -DKOKKOS_ENABLE_CUDA_LAMBDA --expt-extended-lambda -I./ -I/home/brad/opt/kokkos-openmp-cuda/include -I/opt/cuda-9.0/include -L/home/brad/opt/kokkos-openmp-cuda/lib -L/opt/cuda-9.0/lib64 -fopenmp  -lkokkos -ldl -lcudart -lcuda --std=c++11 -arch=sm_50 -Xcompiler -fopenmp -O3 some_code_file.cc -o some_code_file.x

-----------------------------------------------------------------------------------------------------------------------------



