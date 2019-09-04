export I_MPI_PIN=enable
export I_MPI_PIN_DOMAIN=1
export I_MPI_PIN_ORDER=range
export I_MPI_DEBUG=5

mpirun -np 16 ./sus.arches-no-kokkos char-ox-0016-16.ups | tee cpu-mpi-only-char-ox-16cp.txt
mpirun -np 16 ./sus.arches-no-kokkos char-ox-0016-32.ups | tee cpu-mpi-only-char-ox-32cp.txt
mpirun -np 16 ./sus.arches-no-kokkos char-ox-0016-64.ups | tee cpu-mpi-only-char-ox-64cp.txt

