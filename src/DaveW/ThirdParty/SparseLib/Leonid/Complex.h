#ifndef COMPLEX_H
#define COMPLEX_H 1

#include<math.h>
#include<iostream>

//using namespace std;

//Complex Class Definitions
//-------------------------------------------------------------------------
class Complex{
private:
  double a;
  double b;
  
public:
  inline Complex(): a(0), b(0) {}
  inline Complex(double a) : a(a), b(0) {}
  inline Complex(double a, double b): a(a), b(b) {}
  inline Complex (const Complex &C): a(C.a), b(C.b) {}
  inline Complex &operator= (const Complex &C){a = C.a; b=C.b; return(*this);}
  inline Complex &operator= (double x){a = x; b = 0; return(*this);}
  
  inline double abs() {return sqrt(a*a + b*b);}
  double arg();
  inline double &Re(){return a;}
  inline double &Im(){return b;}  
  inline void set(double aa,double bb){a = aa; b = bb;} 
  inline Complex conj(){Complex C; C.a=a; C.b=-b;return(C);}

  
  Complex operator+ (const Complex&) const;
  Complex &operator+=(const Complex&);
  Complex operator- (const Complex&) const;
  Complex &operator-= (const Complex&); 
  Complex operator* (const Complex&) const;
  Complex operator/ (const Complex&) const;
  Complex operator* (double) const;
  Complex operator/ (double) const;
  friend  Complex operator* (double , Complex&); 
  friend  Complex operator/ (double , Complex&);
  
  friend std::ostream &operator<<(std::ostream &output, Complex&);
  friend std::istream &operator>>(std::istream &input, Complex&);
};

#endif
