#ifndef cSMATRIX_H
#define ccSMatrix_H 1

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

#include <iostream.h>
#include <fstream.h>
#include <CoreDatatypes/cMatrix.h>
#include <Math/Complex.h>
#include <CoreDatatypes/cVector.h>

namespace SCICore {
namespace CoreDatatypes {

class cSMatrix:public cMatrix{
  
private:
  Complex *a;
  int *row_ptr;
  int *col;
  int nrows;
  int ncols;
  int nnz;
  
public:   
 
  cSMatrix(int nrows, int ncols,int nnz,Complex *a, int * row, int *col );
  ~cSMatrix();
    
  friend ostream &operator<< (ostream &output, cSMatrix &B);
  
 
 cVector  operator*(cVector &V);

 void mult(cVector& V,cVector& tmp);
 virtual Complex& get(int row, int col);

};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:34  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:01  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:52  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:24  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif





