#ifndef REG_TOOLS_H
#define REG_TOOLS_H 1

#include "MatrixDense.h"
#include "Vector.h"
#include "LinearSystem.h"
#include "SVD.h"

// truncated SVD
void  tsvd(MatrixDense<double>* A, ZVector<double>* b, ZVector<double> *x,
	   double truncate);

// dumped SVD 
void  dsvd(MatrixDense<double>* A, ZVector<double>* b, ZVector<double> *x,
	   double lambda);

// tikhonov
void  tikhonov(MatrixDense<double>* A, ZVector<double>* b, ZVector<double> *x,
	   double lambda);
#endif
