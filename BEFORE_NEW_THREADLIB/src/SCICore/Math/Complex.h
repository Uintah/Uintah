
/*
 *  Complex.h:  Complex numbers
 *
 *  Written by:
 *   Leonid Zhukov
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */


#ifndef COMPLEX_H
#define COMPLEX_H 1

#include <SCICore/share/share.h>

#include<math.h>

// KCC stuff
#include <iostream.h>
//class ostream;
//class istream;

namespace SCICore {
namespace Math {

class SCICORESHARE Complex{
private:
  double a;
  double b;
  
public:
  Complex(): a(0), b(0) {}
  Complex(double a, double b): a(a), b(b) {}
  Complex (const Complex &C): a(C.a), b(C.b) {}
  Complex &operator= (const Complex &C){a = C.a; b=C.b; return(*this);}
  Complex &operator= (double x){a = x; b = x; return(*this);}
  
  double abs() {return sqrt(a*a + b*b);}
  double arg();
  double &Re(){return a;}
  double &Im(){return b;}  
  void set(double aa,double bb){a = aa; b = bb;} 
  Complex conj(){Complex C; C.a=a; C.b=-b;return(C);}

  
  Complex operator+ (const Complex&) const;
  Complex operator- (const Complex&) const;
  Complex operator* (const Complex&) const;
  Complex operator/ (const Complex&) const;
  Complex operator* (double) const;
  Complex operator/ (double) const;
  friend  SCICORESHARE Complex operator* (double , Complex&); 
  friend  SCICORESHARE Complex operator/ (double , Complex&);
  
  friend SCICORESHARE ostream &operator<<(ostream &output, Complex&);
  friend SCICORESHARE istream &operator>>(istream &input, Complex&);
};

} // End namespace Math
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:02  mcq
// Initial commit
//
// Revision 1.4  1999/07/01 16:44:22  moulding
// added SHARE to enable win32 shared libraries (dll's)
//
// Revision 1.3  1999/05/06 19:56:18  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:22  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//

#endif

