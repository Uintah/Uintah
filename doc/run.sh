export OMP_NESTED=true
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=spread,spread
export OMP_PLACES=threads

mpirun -np 1 ./sus.kokkos-host-O2 -npartitions 2 RMCRT_bm1_DO.ups
