
/*
 *  DenseMatrix.h:  Dense matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DenseMatrix_h
#define SCI_project_DenseMatrix_h 1

#include <Datatypes/Matrix.h>

class DenseMatrix : public Matrix {
    int nc;
    int nr;
    double** data;
    double* dataptr;
public:
    DenseMatrix(int, int);
    virtual ~DenseMatrix();
    DenseMatrix(const DenseMatrix&);
    DenseMatrix& operator=(const DenseMatrix&);
    virtual double& get(int, int);
    virtual void put(int, int, const double&);
    virtual int nrows();
    virtual int ncols();
    virtual void solve(ColumnMatrix&);
    virtual void zero();

    virtual void mult(ColumnMatrix& product, ColumnMatrix& multiplier,
		      int b=-1, int e=-1);
    virtual void mult_transpose(ColumnMatrix& product, ColumnMatrix& multiplier,
				int b=-1, int e=-1);
    virtual void print();

    MatrixRow operator[](int r);
};

#endif
