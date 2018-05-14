This contains some build instruction notes for getting up and running on this Kokkos dev branch.  

Grab Kokkos (I put mine in my ~/src directory)

git clone https://github.com/kokkos/kokkos.git ~/src/

Make a Kokkos folder (I put mine in ~/opt)
mkdir ~/opt/kokkos-openmp
mkdir ~/opt/kokkos-openmp/build
cd ~/opt/kokkos-openmp/build

-----------------------------------------------------------------------------------------------------------------------------
Here is how I configure Kokkos for OpenMP

~/src/kokkos/generate_makefile.bash --kokkos-path=/home/brad/src/kokkos --prefix=/home/brad/opt/kokkos-openmp --with-openmp

Note: You must have nvcc in your path for this to work! Specifying the path to the compiler will over-ride use of the nvcc_wrapper which will cause compilation failure.

Here is how I configure Kokkos for CUDA
1) Apply my Kokkos patch for asynchronous, I have placed it in this Uintah branch's src directory. 
git apply ~/kokkos_brad_dec22.patch (works from anywhere inside the kokkos source tree)
2) Edit Kokkos's Makefile script, its bugged: 
vim ~/src/kokkos/core/src/Makefile

Remove the ? on line 11 so it reads:
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
else
  CXX ?= g++
endif

Here is how I configure Kokkos for OpenMP and 
~/src/kokkos/generate_makefile.bash --kokkos-path=/home/brad/src/kokkos --prefix=/home/brad/opt/kokkos-openmp-cuda-9.0-gcc-6.4 --with-openmp --with-cuda=/home/brad/opt/cuda-9.0 --arch=Maxwell52
-----------------------------------------------------------------------------------------------------------------------------

make
make install

Your Kokkos should now be built and ready to go.
-----------------------------------------------------------------------------------------------------------------------------

For a CUDA build, you must edit nvcc_wrapper whereever Kokkos was installed and edit a couple things:

vim ~/opt/kokkos-openmp-cuda-9.0-gcc-6.4/bin/nvcc_wrapper

1) Put in the correct default_arch (e.g. default_arch="sm_50")
2) Comment out depfile arguments, they're broken right now:

  # Handle depfile arguments.  We map them to a separate call to nvcc.
#  -MD|-MMD)
#    depfile_separate=1
#    host_only_args="$host_only_args $1"
#    ;;
#  -MF)
#    depfile_output_arg="-o $2"
#    host_only_args="$host_only_args $1 $2"
#    shift
#    ;;
#  -MT)
#    depfile_target_arg="$1 $2"
#    host_only_args="$host_only_args $1 $2"
#    shift
#    ;;

-----------------------------------------------------------------------------------------------------------------------------


Then in Uintah, just add this line to your configure script:

--with-kokkos=/home/brad/opt/kokkos-openmp

It seems also this is needed: LDFLAGS='-ldl'

If you are using CUDA, you must put the nvcc_wrapper for the CXX
CXX=/home/brad/opt/kokkos-cuda-9.0-gcc-6.4/bin/nvcc_wrapper 

Note that we are requiring OpenMP for all builds, CUDA and non-CUDA.  

For an example of building a single Kokkos test program with both CUDA and OpenMP support:

/home/brad/opt/kokkos-cuda-9.0-gcc-6.4/bin/nvcc_wrapper -DKOKKOS_ENABLE_CUDA_LAMBDA --expt-extended-lambda -I./ -I/home/brad/opt/kokkos-openmp-cuda-9.0-gcc-6.4/include -I/opt/cuda-9.0/include -L/home/brad/opt/kokkos-openmp-cuda-9.0-gcc-6.4/lib -L/opt/cuda-9.0/lib64 -fopenmp  -lkokkos -ldl -lcudart -lcuda --std=c++11 -arch=sm_52 -Xcompiler -fopenmp -O3 team_demo.cc -o team_demo.x

-----------------------------------------------------------------------------------------------------------------------------
When you build Uintah, make sure you specify sus: "make -j8 sus" and not just "make -j8", the kokkos build fails on all Uintah stuff. 

Both the MPI scheduler and the Unified Scheduler work.  As a reminder, -nthreads triggers the scheduler

MPI scheduler: sus RMCRT_bm1_DO.ups
Unified scheduler: sus -nthreads 1 RMCRT_bm1_DO.ups

For CPU/Xeon Phi tasks, because of how Kokkos+OpenMP runs loops, both schedulers will use all cores on an MPI rank anyway.
For GPU tasks, you must use the Unified Scheduler and you must use at least 2 threads, and often it's a good idea to use all available threads.
-------------------------------------------



