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


#include<math.h>

class Complex{
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
  friend  Complex operator* (double , Complex&); 
  friend  Complex operator/ (double , Complex&);
  
  friend ostream &operator<<(ostream &output, Complex&);
  friend istream &operator>>(istream &input, Complex&);
};



#endif








