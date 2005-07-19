
#include <mpi.h>
#include <stdio.h>

int
main( int argc, char *argv[] )
{
  printf("hello\n");

  MPI_Init( &argc, &argv );

  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  printf("goodbye\n");


  MPI_Finalize();

}
