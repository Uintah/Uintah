
/*
 *  TriDiagonalMatrix.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef DATATYPES_TRIDIAGONALMATRIX_H
#define DATATYPES_TRIDIAGONALMATRIX_H 1

#include <SCICore/Datatypes/Matrix.h>

namespace SCICore {
namespace Datatypes {

typedef double TriRow[3];

class SCICORESHARE TriDiagonalMatrix : public Matrix {
    int rows;
    TriRow* data;
public:
    TriDiagonalMatrix(int nrows);
    ~TriDiagonalMatrix();
    void solve(ColumnMatrix&);
    void setrow(int i, double l, double m, double r);


    virtual double& get(int, int);
    virtual void zero();
    virtual int nrows() const;
    virtual int ncols() const;
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& v);
    virtual double minValue();
    virtual double maxValue();
    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1, 
		      int spVec=0) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, 
				int end=-1, int spVec=0);
};

} // End namespace Datatypes
} // End namespace SCICore


#endif
