
//
//  A simple test program which can give some confidence that lapack is installed and running
//  correctly on a system.
//
//  Note, the "#include <mkl_lapack.h>" may need to be changed to point to the flavor of lapack
//  that exists on a given system.
//
//
//  On TACC Ranger:  (using mkl)
//
//        icc -I/scratch/projects/tg/uintah/SystemLibLinks/mkl/include test_lapack.c \
//            -L/scratch/projects/tg/uintah/SystemLibLinks/mkl/lib \
//                -lmkl -lmkl_core -lmkl_intel_lp64 -lguide -lpthread \
//            -Wl,-rpath -Wl,/scratch/projects/tg/uintah/SystemLibLinks/mkl/lib
//
//        /scratch/projects/tg/uintah/SystemLibLinks/mkl/ =>
//            include -> /opt/apps/intel/mkl/10.0.1.014/include
//            lib -> /opt/apps/intel/mkl/10.0.1.014/lib/em64t
//
//  On CHPC delicatearch: (using acml)
//
//        g++ -I/uufs/arches/sys/pkg/acml/3.5.0/gnu64/include test_lapack.c \
//            -L/uufs/arches/sys/pkg/acml/3.5.0/gnu64/lib -lacml -lacml_mv -lg2c \
//            -Wl,-rpath -Wl,/uufs/arches/sys/pkg/acml/3.5.0/gnu64/lib

/*
Results from Ranger:

login3% ./a.out
   1.000000     -1.000000     2.000000     -1.000000  
   2.000000     -2.000000     3.000000     -3.000000  
   1.000000     1.000000     1.000000     0.000000  
   1.000000     -1.000000     4.000000     3.000000  
info 0 
   -7.000000 
   3.000000 
   2.000000 
   2.000000 
*/


#include <stdlib.h>
#include <stdio.h>

#define DELICATEARCH

#if defined( RANGER )
#  include <mkl_lapack.h>
#elif defined( DELICATEARCH )
#  include <acml.h>
#else
// Not sure what the correct default lapack.h is...
#  include <lapack.h>
#endif

#include <math.h>

#define NDIM 4

int
main ()
{
  int N, NRHS, LDA, LDB;
  static int IPIV[NDIM], INFO;

  double *  A = (double*) malloc(NDIM*NDIM*sizeof(double));
  double * B = (double*) malloc(NDIM*sizeof(double));

  N    = NDIM; 
  LDA  = NDIM;
  LDB  = NDIM;

  NRHS = 1;

  A[0]  =  1.0;
  A[4]  = -1.0;
  A[8]  =  2.0;
  A[12] = -1.0;
  A[1]  =  2.0;
  A[5]  = -2.0;
  A[9]  =  3.0;
  A[13] = -3.0;
  A[2]  =  1.0;
  A[6]  =  1.0;
  A[10] =  1.0;
  A[14] =  0.0;
  A[3]  =  1.0;
  A[7]  = -1.0;
  A[11] =  4.0;
  A[15] =  3.0;

  for( int ii = 0; ii < N; ii++) {
    for( int jj = 0; jj < N; jj++) {
      printf( "   %f  ", A[ ii + (N*jj) ] );
    }
    printf( "\n" );
  }

  B[0] = -8.0;
  B[1] = -20.0;
  B[2] = -2.0;
  B[3] = 4.0;

  dgesv_(&N, &NRHS, A, &LDA, (int*)&IPIV, B, &LDB, &INFO);

  printf("info %d \n",INFO);

  for( int idx = 0 ; idx < N; idx++ ) {
    printf( "   %f \n", B[idx] );
  }
}

