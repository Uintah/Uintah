
/*
 *  cSMatrix.h : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef CSMATRIX_H
#define CSMATRIX_H 1

#include <Core/share/share.h>
#include <Core/Datatypes/cMatrix.h>
#include <Core/Datatypes/cVector.h>

namespace SCIRun {

class SCICORESHARE cSMatrix:public cMatrix{
  
private:
    typedef std::complex<double> Complex;
  Complex *a;
  int *row_ptr;
  int *col;
  int nrows;
  int ncols;
  int nnz;
  
public:   
 
  cSMatrix(int nrows, int ncols,int nnz,Complex *a, int * row, int *col );
  ~cSMatrix();
    
  friend SCICORESHARE std::ostream &operator<< (std::ostream &output, cSMatrix &B);
  
 
 cVector  operator*(cVector &V);

 void mult(cVector& V,cVector& tmp);
 virtual Complex& get(int row, int col);

};

} // End namespace SCIRun


#endif





