//static char *id="@(#) $Id$";

/*
 *  Complex.cc:  Complex numbers
 *
 *  Written by:
 *   Leonid Zhukov
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include"Complex.h"

// Moved to Complex.h for KCC...
//#include<iostream.h>

#ifndef M_PI
#define M_PI 3.1415926535898
#endif

#ifndef M_PI
#define M_PI 3.1415926535898
#endif

namespace SCICore {
namespace Math {

//------------------------------------------------------------------
double Complex::arg(){
  if(a!=0) return(atan(b/a));
  else if((a==0)&&(b==0)) return (0);
  else return(M_PI/2);
}

//------------------------------------------------------------------
Complex Complex::operator+(const Complex& C2) const{
  Complex C;

  C.a = a + C2.a;
  C.b = b + C2.b;

  return(C);
}

//------------------------------------------------------------------
Complex Complex::operator-(const Complex& C2) const{
  Complex C;

  C.a = a - C2.a;
  C.b = b - C2.b;

  return(C);
}
 
//------------------------------------------------------------------
Complex Complex::operator*(const Complex& C2) const{
  Complex C;

  C.a = a*C2.a - b*C2.b;
  C.b = b*C2.a + a*C2.b;

  return(C);
}

//------------------------------------------------------------------
Complex Complex::operator*(double t) const{
  Complex C;
  
  C.a = t*a;
  C.b = t*b;
  
  return(C);
}

//------------------------------------------------------------------
Complex Complex::operator/(double t) const{
  Complex C;
  
  C.a = a/t;
  C.b = b/t;
  return(C);
}

//------------------------------------------------------------------
Complex Complex::operator/(const Complex &C2) const{
  Complex C;
  double d = C2.a*C2.a + C2.b*C2.b;
  
  C.a = (a*C2.a + b*C2.b)/d;
  C.b = (b*C2.a - a*C2.b)/d;
  return(C);
}

//------------------------------------------------------------------
Complex  operator*(double t, Complex &C1){
  Complex C;

  C.a = t*C1.a;
  C.b = t*C1.b;

  return(C);
}

//------------------------------------------------------------------
Complex  operator/(double t,Complex &C1){
  Complex C;  
  double d = C1.a*C1.a + C1.b*C1.b;
 
  C.a = t*C1.a/d;
  C.b = -t*C1.b/d;
  return(C);
} 
  
//------------------------------------------------------------------
ostream &operator<< (ostream &output, Complex &C){

  if(C.b>=0)
    output<<"("<<C.a<<" + "<<C.b<<"*i)";
  else
    output<<"("<<C.a<<" - "<<-C.b<<"*i)";

    
//  output<<C.a<<" "<<C.b<<" ";

    return(output);
}

//------------------------------------------------------------------
  
istream &operator>> (istream &input, Complex &C){
 
    input>>C.a;
    input>>C.b;
    
  return(input);
}

//------------------------------------------------------------------  

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
// Revision 1.2  1999/07/01 16:48:21  moulding
// added #defines to make win32 work
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//
