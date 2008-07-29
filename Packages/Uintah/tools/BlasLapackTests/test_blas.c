
//
//  Note, the "#include <mkl_blas.h>" may have to be adjusted to the
//  correct include file.  Also, the "mkl_" may need to be modified
//  throughout the rest of the program.
//
//  On TACC Ranger:
//
//     icc -I/scratch/projects/tg/uintah/SystemLibLinks/mkl/include test_blas.c \
//         -L/scratch/projects/tg/uintah/SystemLibLinks/mkl/lib -lmkl -lguide -lpthread \
//         -Wl,-rpath -Wl,/scratch/projects/tg/uintah/SystemLibLinks/mkl/lib
//
//     Note, while this shows compilation, this code actually segfaults... I'll try
//     to figure out why and add in what the right answer should look like...
//

#include <stdlib.h>

#include <mkl_cblas.h>

int
main()
{
  int dim1 = 1, dim2 = 1, dim3 = 1;
  int alpha = 0, beta = 0;
  const double * a = NULL, * b = NULL;
  double * c = NULL;

  cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, 
               dim1, dim3, dim2, alpha, a, dim1, b, dim2, beta, c, dim1 );

  return 0;
}
