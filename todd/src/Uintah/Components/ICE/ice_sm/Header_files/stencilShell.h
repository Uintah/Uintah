
/*
   Context and functions needed for shell matrix.
*/

#ifndef STENCIL_SHELL_H
#define STENCIL_SHELL_H

#include "sles.h"

#define FORT_STENCILMULT stencilmult_

typedef struct {
   Vec as;
   Vec aw;
   Vec ap;
   Vec ae;
   Vec an;
   int m;
   int n;
} stencilMatrix;

extern int stencilMult( Mat A, Vec x, Vec y );
extern int stencilSetValues( Mat A, int row, int* nrows, int col, int* ncols, Scalar* values, InsertMode mode );
extern int defineStencilOperator( int m, int n, stencilMatrix* stencil, Mat* A );
extern void FORT_STENCILMULT( Scalar*, Scalar*, Scalar*, Scalar*,Scalar*, Scalar*, Scalar*, int*, int* );
 
#endif

