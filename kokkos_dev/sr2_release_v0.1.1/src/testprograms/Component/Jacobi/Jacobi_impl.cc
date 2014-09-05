/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  Jacobi_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   October, 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <testprograms/Component/Jacobi/Jacobi_impl.h>
#include <Core/Util/NotFinished.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

using namespace Jacobi_ns;
using namespace std;

// Solve Laplace equation using Jacobi iteration method
// Kostadin Damevski, University of Utah, 2002

/*****************************************************************************/
#define NUMITER   1000000                /* Max num of Iterations            */
#define MSG_TAG_DOWN     100             /* MsgTag for messages down         */
#define MSG_TAG_UP       101             /* MsgTag for messages up           */
#define ROOT     0                       /* Root PE                          */
#define MAX(x,y) ( ((x) > (y)) ? x : y ) /* Define Max Function              */
#define stoptol  0.5E-2                  /* stopping tolerance: Convergence  */
/*****************************************************************************/


Jacobi_impl::Jacobi_impl()
{
}

Jacobi_impl::~Jacobi_impl()
{
}

int Jacobi_impl::solveHeatEquation(SSIDL::array2<double>& arr, double top, double bottom,
				   double left, double right)
{
  int        size;                    /* Actual Number of PEs       */
  int        rank;                    /* My PE number               */
  int        nstop;                   /* Interation Stop Parameter  */

  SSIDL::array2<double> t;  
  SSIDL::array2<double> told;           /* Temp. Arrays: Old & New */
  double     dt;                      /* Delta t */
  double     dtg    ;                 /* Delta t global */
  int        i, j, k, l;
  int        iter;                    /* Current Number of Iterations */
  int        n_cols,n_rows;           /* Number of Cols and Rows */

  MPI_Comm_size(MPI_COMM_WORLD, &size );
  MPI_Comm_rank(MPI_COMM_WORLD, &rank );
  MPI_Status status;         
  n_cols = arr.size2();
  n_rows = arr.size1();
  nstop = 0;

  /*Resize array:*/
  told.resize(arr.size1()+2,arr.size2());
  t.resize(arr.size1()+2,arr.size2());

  /* Copy arr into t */
  for( i=1; i<n_rows; i++ )       
    for( j=0; j<n_cols; j++ )
      t[i][j] = arr[i-1][j];
  
  /* Set the Boundary Conditions in t */
  if( rank == 0 )
    for( j=0; j<n_cols; j++ )
      t[0][j] = top;   //Top boundary

  if( rank == size-1 )
    for( j=0; j<n_cols; j++ )
      t[n_rows+1][j] = bottom;  //Bottom boundary

  for( i=0; i<n_rows+2; i++ ) {
    t[i][0]    = left; //Set Left Boundary
    t[i][n_cols-1] = right; //Set Right Boundary
  }

  /* Copy t into told */
  for( i=0; i<n_rows+2; i++ )       
    for( j=0; j<n_cols; j++ )
      told[i][j] = t[i][j];
  
 
  for( iter=1; iter < NUMITER; iter++ ) { /* Start of Big Iteration Loop */
    for( i=1; i<n_rows+1; i++ )
      for( j=1; j<n_cols; j++ )
        t[i][j] = 0.25 * (told[i+1][j] + told[i-1][j] 
			     + told[i][j+1] + told[i][j-1]);
    
    dt = 0.;
    for( i=1; i<n_rows+1; i++ )       
      for( j=1; j<n_cols; j++ ){
        dt = MAX(fabs(t[i][j]-told[i][j]), dt);
        told[i][j] = t[i][j];
      }

    /*   Sending Down; Only size-1 do this           */    
    if( rank < size-1 )    
      MPI_Send( &t[n_rows+1][0], n_cols, MPI_DOUBLE, rank+1, MSG_TAG_DOWN, MPI_COMM_WORLD);

    /*   Receive  Msg MSG_TAG_DOWN rank-1 above              */    
    if( rank != 0 )        
      MPI_Recv( &t[0][0], n_cols, MPI_DOUBLE, MPI_ANY_SOURCE, MSG_TAG_DOWN, MPI_COMM_WORLD, &status);

    /*   Sending Msg Up  ; Only rank+1 does this     */    
    if( rank != 0 )        
      MPI_Send( &t[1][0], n_cols, MPI_DOUBLE, rank-1, MSG_TAG_UP, MPI_COMM_WORLD);

    /*   Receive Msg Up from rank+1 below       */    
    if( rank != size-1 )   
      MPI_Recv( &t[n_rows][0],n_cols, MPI_DOUBLE, MPI_ANY_SOURCE, MSG_TAG_UP, MPI_COMM_WORLD, &status);
    
    MPI_Reduce(&dt, &dtg, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    
    if( rank == 0 ) {
      if( (iter%100) == 0 ) {
	printf("\nRANK = %4d; ITER = %5d; GlobalMaxAbsChange:dtg = %10.3e\n", 
	       rank, iter, dtg); 
      }
      if(dtg < stoptol) nstop = 1;
    }
    
    MPI_Bcast(&nstop, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    MPI_Barrier( MPI_COMM_WORLD );
    if(nstop == 1) break;    
  }  /* End of Big Iteration Loop */

  /* Copy t's result into arr*/
  for( i=0; i<n_rows; i++ )       
    for( j=0; j<n_cols; j++ )
      arr[i][j] = t[i+1][j];

  
  //printf("\nSample Output: RANK =%4d; ITER =%5d\n J/I ",rank, iter);
  for( l=0; l<=9; l++ ){
    //printf("\n%4d ",l);
    for( k=0; k<=5; k++ ){
      //printf("%7.2f",t[l][k]);
    }
  }
  
  return 0;
}





