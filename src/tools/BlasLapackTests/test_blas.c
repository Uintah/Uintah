
//
//  Note, the "#include <mkl_blas.h>" may have to be adjusted to the
//  correct include file.  Also, the "mkl_" may need to be modified
//  throughout the rest of the program.
//
//  On TACC Ranger: (using mkl)
//
//     icc -I/scratch/projects/tg/uintah/SystemLibLinks/mkl/include test_blas.c \
//         -L/scratch/projects/tg/uintah/SystemLibLinks/mkl/lib -lmkl -lguide -lpthread \
//         -Wl,-rpath -Wl,/scratch/projects/tg/uintah/SystemLibLinks/mkl/lib
//
//  On CHPC DelicateARch: (using acml)
//
//     g++ -I/uufs/arches/sys/pkg/acml/3.5.0/gnu64/include test_blas.c \
//         -L/uufs/arches/sys/pkg/acml/3.5.0/gnu64/lib -lacml -lg2c \
//         -Wl,-rpath -Wl,/uufs/arches/sys/pkg/acml/3.5.0/gnu64/lib
//

/*
  Results:

 -2.0
  0.0
 -7.0

*/

#include <stdlib.h>
#include <stdio.h>

#define RANGER

#if defined( RANGER )
#  include <mkl_blas.h>
#elif defined( DELICATEARCH )
#  include <acml.h>
#else
#  include <cblas.h>
#endif

double m[] = {
  3, 1, 3,        
  1, 5, 9,
  2, 6, 5
};

double x[] = {
  -1, -1, 1
};

double y[] = {
  0, 0, 0
};

int
main()
{
  int i, j;
  
  DGEMV('N', 3, 3, 1.0, m, 3, x, 1, 0.0, y, 1);
  
  for (i=0; i<3; ++i) {
    printf( "%5.1f\n", y[i] );
  }
  
  return 0;
}

