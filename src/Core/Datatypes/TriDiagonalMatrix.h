
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

//
// $Log$
// Revision 1.4  2000/07/12 15:45:11  dmw
// Added Yarden's raw output thing to matrices, added neighborhood accessors to meshes, added ScalarFieldRGushort
//
// Revision 1.3  1999/08/25 03:48:44  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif
