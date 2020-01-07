#/bin/sh

../src/configure --enable-debug --enable-all-components --with-boost=/usr --with-mpi=built-in --with-hypre-include=/usr/include/hypre --with-hypre-lib=/usr/lib/x86_64-linux-gnu --enable-wasatch_3p CC=mpicc CXX=mpicxx F77=mpif77
