
/*
 *  Complex.h: Complex number support
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Math_Complex_h
#define SCI_Math_Complex_h 1

#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <Math/Trig.h>

class ostream;

class Complex {
    double r;
    double i;
public:
    Complex();
    Complex(double re, double im);
    ~Complex();

    double re() const;
    double im() const;

    Complex operator+(const Complex&) const;
    Complex operator-(const Complex&) const;
    Complex operator-() const;
    Complex operator*(const Complex&) const;
    Complex operator*(double) const;
    Complex operator/(const Complex&) const;
    Complex& operator/=(const Complex&);
    Complex& operator+=(const Complex&);
    friend inline Complex Sqrt(const Complex&);
    friend inline Complex Exp(const Complex&);
    friend inline ostream& operator<<(ostream&, const Complex&);
};

inline Complex::Complex()
{
}

inline Complex::Complex(double r, double i)
: r(r), i(i)
{
}

inline Complex::~Complex()
{
}

double Complex::re() const
{
    return r;
}

double Complex::im() const
{
    return i;
}

inline Complex Complex::operator+(const Complex& c) const
{
    return Complex(r+c.r, i+c.i);
}

inline Complex Complex::operator-(const Complex& c) const
{
    return Complex(r-c.r, i-c.i);
}

inline Complex Complex::operator-() const
{
    return Complex(-r, -i);
}

inline Complex Complex::operator*(const Complex& c) const
{
    return Complex(r*c.r-i*c.i, r*c.i+i*c.r);
}

inline Complex Complex::operator*(double d) const
{
    return Complex(r*d, i*d);
}

inline Complex& Complex::operator+=(const Complex& c)
{
    r+=c.r; i+=c.i;
    return *this;
}

inline Complex Complex::operator/(const Complex& c) const
{
    double di=Abs(c.i);
    double dr=Abs(c.r);
    double d=di+dr;
    double rd=r/d;
    double id=i/d;
    double crd=c.r/d;
    double cid=c.i/d;
    double mag=crd*crd+cid*cid;
    return Complex((rd*crd+id*cid)/mag,
		   (id*crd-rd*cid)/mag);
}

inline Complex& Complex::operator/=(const Complex& c)
{
    double di=Abs(c.i);
    double dr=Abs(c.r);
    double d=di+dr;
    double rd=r/d;
    double id=i/d;
    double crd=c.r/d;
    double cid=c.i/d;
    double mag=crd*crd+cid*cid;
    r=(rd*crd+id*cid)/mag;
    i=(id*crd-rd*cid)/mag;
    return *this;
}

inline Complex Sqrt(const Complex& c)
{
    if(c.r == 0.0 && c.i == 0.0)
	return Complex(0.0, 0.0);
    double s=Sqrt((Abs(c.r) + Hypot(c.r, c.i))*0.5);
    double d=(c.i/s)*0.5;
    if(c.r>0.0)
	return Complex(s,d);
    else if(c.i>= 0.0)
	return Complex(d,s);
    else
	return Complex(-d, -s);
}

inline Complex Exp(const Complex& c)
{
    double er=Exp(c.r);
    return Complex(er*Cos(c.i), er*Sin(c.i));
}

#include <iostream.h>

inline ostream& operator<<(ostream& out, const Complex& c)
{
    out << "(" << c.r << ", " << c.i << ")";
    return out;
}

#endif
