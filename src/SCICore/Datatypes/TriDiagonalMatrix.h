
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

#include <CoreDatatypes/Matrix.h>

namespace SCICore {
namespace CoreDatatypes {

typedef double TriRow[3];

class TriDiagonalMatrix : public Matrix {
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
		      int& flops, int& memrefs, int beg=-1, int end=-1) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, int end=-1);
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:30  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:57  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:48  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:19  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif
