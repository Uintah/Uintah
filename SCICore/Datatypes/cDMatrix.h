
/*
 *  cDMatrix.h : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef CDMATRIX_H
#define CDMATRIX_H 1

#include <SCICore/share/share.h>
#include <SCICore/Datatypes/cMatrix.h>
#include <SCICore/Datatypes/cVector.h>

namespace SCICore {
namespace Datatypes {

class SCICORESHARE cDMatrix:public cMatrix{
  
private:
    typedef std::complex<double> Complex;
    Complex **a;
  int Size;
    
public:

  cDMatrix(int N); //constructor
  cDMatrix(const cDMatrix &B); //copy constructor;
  cDMatrix &operator=(const cDMatrix &B); //assigment operator  
  ~cDMatrix(); //Destructor;
  
  int size() {return Size;};
  Complex &operator()(int i, int j);
  int load(char* filename);
  
  cDMatrix operator+(const cDMatrix& B) const;
  cDMatrix operator-(const cDMatrix& B) const;
  cDMatrix operator*(const cDMatrix& B) const;

  friend SCICORESHARE cDMatrix operator*(const cDMatrix& B,Complex x);
  friend SCICORESHARE cDMatrix operator*(Complex x, const cDMatrix& B);
  
  friend SCICORESHARE cDMatrix operator*(const cDMatrix& B,double x);
  friend SCICORESHARE cDMatrix operator*(double x, const cDMatrix& B);

  
  friend SCICORESHARE std::ostream &operator<< (std::ostream &output, cDMatrix &B);
  
  cVector  operator*(cVector &V);

  void mult(cVector& V,cVector& tmp);
  virtual Complex& get(int row, int col);
  
 };

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.5  1999/10/07 02:07:35  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/08/25 03:48:47  sparker
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
// Revision 1.1  1999/07/27 16:56:33  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:00  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:52  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:23  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif
