/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Datatypes/Matrix.h>

namespace SCIRun {

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

} // End namespace SCIRun


#endif
