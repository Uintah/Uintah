#include "mpi.h"
#include <cmath>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>

using namespace std;

int main(int argc, char *argv[]){

  int my_rank; // rank of process
  int np; // number of processes
  double time1, time2;

  double *testarray = new double[20];
  double *globalarray = new double[20];
  
  for ( int i = 0; i < 20 ; i ++ )
    testarray[i] = 0;
  
  // starting up MPI
  MPI_Init(&argc, &argv);

  MPI_Barrier(MPI_COMM_WORLD);
  
  time1 = MPI_Wtime();
  
  // Find out process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Find out number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int i;

  if ( my_rank == 0 ) {

    for ( i = 0; i < 5; i ++ )
      testarray[i] = i;

  }

  
  if ( my_rank == 1 ) {

    for ( i = 5; i < 10; i ++ )
      testarray[i] = i;

  }

  if ( my_rank == 2 ) {

    for ( i = 10; i < 15; i ++ )
      testarray[i] = i;

  }

  if ( my_rank == 3 ) {

    for ( i = 15; i < 20; i ++ )
      testarray[i] = i;

  }


  if ( my_rank == 0 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "testarray rank 0 [" << i << "] = " << testarray[i] << endl;
  }

  if ( my_rank == 1 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "testarray rank 1 [" << i << "] = " << testarray[i] << endl;
  }

  if ( my_rank == 2 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "testarray rank 2 [" << i << "] = " << testarray[i] << endl;
  }

  if ( my_rank == 3 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "testarray rank 3 [" << i << "] = " << testarray[i] << endl;
  }
  
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;
  
  // put  if ( my_rank == 0 ) here , code wont work
  MPI_Reduce(testarray, globalarray, 20, MPI_DOUBLE, MPI_SUM, 0, comm);

  MPI_Barrier(comm);
  
  time2 = MPI_Wtime();
  
//   if ( my_rank == 0 ) {
//     cout << " time used up (S) = " << time2 - time1 << endl;
//     for ( int ii = 0; ii < 20; ii ++ )
//       cout << "globalarray[" << ii << "] = " << globalarray[ii] << endl;
//   }


  // this will show that only rank 0 get the globalarray values updated
  // as it indicated in MPI_Reduce command.
  if ( my_rank == 0 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "globalarray rank 0 [" << i << "] = " << globalarray[i] << endl;
  }

  if ( my_rank == 1 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "globalarray rank 1 [" << i << "] = " << globalarray[i] << endl;
  }

  if ( my_rank == 2 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "globalarray rank 2 [" << i << "] = " << globalarray[i] << endl;
  }

  if ( my_rank == 3 ) {
    for ( i = 0; i < 20 ; i ++ )
      cout << "globalarray rank 3 [" << i << "] = " << globalarray[i] << endl;
  }
  
  delete[] testarray;
  delete[] globalarray;
  
  MPI_Finalize();

  
  return 0;


}
  

  
