#include <mpi.h>
#include <iostream

int
main( int argc, char *argv[] )
{
  cout << "hello\n";

  MPI_Init( &argc, &argv );

  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  cout << "goodbye\n";

  MPI_Finalize();

  return 0;
}
