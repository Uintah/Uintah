/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  LUFactor_impl.cc: Test client for PIDL
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   October, 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <testprograms/Component/LUFactor/LUFactor_impl.h>
#include <Core/Util/NotFinished.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

#define DEBUGRANK 0
#define DEBUG

using namespace LUFactor_ns;
using namespace std;

// Kostadin Damevski, University of Utah, 2002

LUFactor_impl::LUFactor_impl()
{
}

LUFactor_impl::~LUFactor_impl()
{
}

int LUFactor_impl::LUFactorize(const SSIDL::array2<double>& A)
{

  int SIZE = A.size1();
  int size; //number of MPI processes
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size );
  MPI_Comm_rank(MPI_COMM_WORLD, &rank );


  for(int j=0;j<SIZE;j++) {
    
    if((j % size) == rank) {
      for(int i=(j+1);i<SIZE;i++) { 
#ifdef DEBUG
	if (rank == DEBUGRANK)
	  cout << " A[" << i << "][" << j << "] = A[" << i 
	       << "][" << j << "] / A[" << j << "][" << j << "]\n";
#endif
	A[i][j] = A[i][j] / A[j][j];
      }
    }


    MPI_Datatype Column;
    MPI_Type_vector(SIZE, 1, SIZE, MPI_FLOAT, &Column);      
    MPI_Type_commit(&Column);
    MPI_Bcast((void*) &(A[0][j]), 1, Column, (j % size), MPI_COMM_WORLD);
    
#ifdef DEBUG
    if (rank == DEBUGRANK) {    
      cout << "Intermediary MATRIX A: ************************\n";
      for(int i=0;i<SIZE;i++) {
	for(int j=0;j<SIZE;j++) 
	  cout << A[i][j] << " ";
	cout << "\n";
      }
      cout << "********************************************\n\n";
    }
#endif 
    
    
    for(int i=(j+1);i<SIZE;i++) {
      for(int k=(j+1);k<SIZE;k++) { 
	if((k % size) == rank) {
#ifdef DEBUG
	  if (rank == DEBUGRANK) 
	    cout << "A[" << i << "][" << k << "] = A[" << i << "][" << k << "] - A[" << j 
		 << "][" << k << "] * A[" << i << "][" << j << "]\n";
#endif
	  A[i][k] = A[i][k] - A[j][k] * A[i][j];
	} 
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }  

#ifdef DEBUG  
  for(int l=0; l<SIZE; l++){
    printf("\n%4d ",l);
    for(int k=0; k<SIZE; k++){
      printf("%7.2f",A[l][k]);
    }
  }
  printf("\nDone\n");
  fflush(stdout);

#endif
  
 
  return 0;
}





