
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

#include <SCICore/share/share.h>
class ostream;
#include <SCICore/Datatypes/cMatrix.h>
#include <SCICore/Math/Complex.h>
#include <SCICore/Datatypes/cVector.h>

namespace SCICore {
namespace Datatypes {

class SCICORESHARE cSMatrix:public cMatrix{
  
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
    
  friend SCICORESHARE ostream &operator<< (ostream &output, cSMatrix &B);
  
 
 cVector  operator*(cVector &V);

 void mult(cVector& V,cVector& tmp);
 virtual Complex& get(int row, int col);

};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/25 03:48:48  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/19 23:52:59  sparker
// Removed extraneous includes of iostream.h  Fixed a few NotFinished.h
// problems.  May have broken KCC support.
//
// Revision 1.2  1999/08/17 06:39:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif





