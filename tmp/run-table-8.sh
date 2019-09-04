mpirun -np 1 ./sus.arches-no-kokkos -nthreads 16 rmcrt-128-16.ups | tee cpu-cpu-rmcrt-1ht-16cp.txt
mpirun -np 1 ./sus.arches-no-kokkos -nthreads 16 rmcrt-128-32.ups | tee cpu-cpu-rmcrt-1ht-32cp.txt
mpirun -np 1 ./sus.arches-no-kokkos -nthreads 16 rmcrt-128-64.ups | tee cpu-cpu-rmcrt-1ht-64cp.txt


mpirun -np 1 ./sus.arches-no-kokkos -nthreads 32 rmcrt-128-16.ups | tee cpu-cpu-rmcrt-2ht-16cp.txt
mpirun -np 1 ./sus.arches-no-kokkos -nthreads 32 rmcrt-128-32.ups | tee cpu-cpu-rmcrt-2ht-32cp.txt
mpirun -np 1 ./sus.arches-no-kokkos -nthreads 32 rmcrt-128-64.ups | tee cpu-cpu-rmcrt-2ht-64cp.txt

export OMP_NUM_THREADS=16
export OMP_NESTED=true
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 1 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-1ht-16cp-16p-01tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 2 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-1ht-16cp-08p-02tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 4 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-1ht-16cp-04p-04tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 8 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-1ht-16cp-02p-08tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 16 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-1ht-16cp-01p-16tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 1 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-1ht-32cp-16p-01tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 2 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-1ht-32cp-08p-02tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 4 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-1ht-32cp-04p-04tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 8 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-1ht-32cp-02p-08tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 16 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-1ht-32cp-01p-16tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 1 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-1ht-64cp-16p-01tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 2 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-1ht-64cp-08p-02tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 4 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-1ht-64cp-04p-04tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 8 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-1ht-64cp-02p-08tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 16 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-1ht-64cp-01p-16tpp.txt

export OMP_NUM_THREADS=32
export OMP_NESTED=true
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 32 -nthreadsperpartition 1 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-2ht-16cp-32p-01tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 2 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-2ht-16cp-16p-02tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 4 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-2ht-16cp-08p-04tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 8 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-2ht-16cp-04p-08tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 16 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-2ht-16cp-02p-16tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 32 rmcrt-128-16.ups | tee cpu-kokkos-rmcrt-2ht-16cp-01p-32tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 32 -nthreadsperpartition 1 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-2ht-32cp-32p-01tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 2 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-2ht-32cp-16p-02tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 4 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-2ht-32cp-08p-04tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 8 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-2ht-32cp-04p-08tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 16 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-2ht-32cp-02p-16tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 32 rmcrt-128-32.ups | tee cpu-kokkos-rmcrt-2ht-32cp-01p-32tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 32 -nthreadsperpartition 1 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-2ht-64cp-32p-01tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 2 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-2ht-64cp-16p-02tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 4 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-2ht-64cp-08p-04tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 8 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-2ht-64cp-04p-08tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 16 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-2ht-64cp-02p-16tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 32 rmcrt-128-64.ups | tee cpu-kokkos-rmcrt-2ht-64cp-01p-32tpp.txt
