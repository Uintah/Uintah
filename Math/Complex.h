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
  
  double abs() {return sqrt(a*a + b*b);}
  double arg();
  double &Re(){return a;}
  double &Im(){return b;}  
  
  Complex operator+ (const Complex&) const;
  Complex operator- (const Complex&) const;
  Complex operator* (const Complex&) const;
  Complex operator/ (const Complex&) const;
  Complex operator* (double) const;
  Complex operator/ (double) const;
  friend  Complex operator* (double , const Complex); 
  friend  Complex operator/ (double , const Complex);
  
  friend ostream &operator<<(ostream &output, const Complex);
  
};











