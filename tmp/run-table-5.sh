export OMP_NUM_THREADS=64
export OMP_NESTED=true
export OMP_PROC_BIND=spread
export OMP_PLACES=threads


mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 1 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-1ht-16cp-064p-001tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 32 -nthreadsperpartition 2 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-1ht-16cp-032p-002tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 4 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-1ht-16cp-016p-004tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 8 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-1ht-16cp-008p-008tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 16 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-1ht-16cp-004p-016tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 32 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-1ht-16cp-002p-032tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 64 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-1ht-16cp-001p-064tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 1 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-1ht-32cp-064p-001tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 32 -nthreadsperpartition 2 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-1ht-32cp-032p-002tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 4 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-1ht-32cp-016p-004tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 8 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-1ht-32cp-008p-008tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 16 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-1ht-32cp-004p-016tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 32 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-1ht-32cp-002p-032tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 64 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-1ht-32cp-001p-064tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 1 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-1ht-64cp-064p-001tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 32 -nthreadsperpartition 2 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-1ht-64cp-032p-002tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 16 -nthreadsperpartition 4 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-1ht-64cp-016p-004tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 8 -nthreadsperpartition 8 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-1ht-64cp-008p-008tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 4 -nthreadsperpartition 16 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-1ht-64cp-004p-016tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 2 -nthreadsperpartition 32 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-1ht-64cp-002p-032tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 64 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-1ht-64cp-001p-064tpp.txt

export OMP_NUM_THREADS=128
export OMP_NESTED=true
export OMP_PROC_BIND=spread
export OMP_PLACES=threads


mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 2 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-2ht-16cp-064p-002tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 128 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-2ht-16cp-001p-128tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 2 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-2ht-32cp-064p-002tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 128 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-2ht-32cp-001p-128tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 2 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-2ht-64cp-064p-002tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 128 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-2ht-64cp-001p-128tpp.txt

export OMP_NUM_THREADS=256
export OMP_NESTED=true
export OMP_PROC_BIND=spread
export OMP_PLACES=threads


mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 4 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-4ht-16cp-064p-004tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 256 char-ox-0064-16.ups | tee mic-kokkos-char-ox-ts-4ht-16cp-001p-256tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 4 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-4ht-32cp-064p-004tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 256 char-ox-0064-32.ups | tee mic-kokkos-char-ox-ts-4ht-32cp-001p-256tpp.txt

mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 64 -nthreadsperpartition 4 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-4ht-64cp-064p-004tpp.txt
mpirun -np 1 ./sus.arches-kokkos-openmp -npartitions 1 -nthreadsperpartition 256 char-ox-0064-64.ups | tee mic-kokkos-char-ox-ts-4ht-64cp-001p-256tpp.txt
