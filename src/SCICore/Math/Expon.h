
/*
 *  Expon.h: Interface to Exponentiation functions...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Math_Expon_h
#define SCI_Math_Expon_h 1

#include <share/share.h>

#include <math.h>

inline SHARE double Pow(double d, double p)
{
    return pow(d,p);
}

inline SHARE int Sqrt(int i)
{
    return (int)sqrt((double)i);
}

inline SHARE double Sqrt(double d)
{
    return sqrt(d);
}

#if defined(linux)||defined(_WIN32)
inline double Cbrt(double d)
{
    return pow(d, 1./3.);
}
#else
inline double Cbrt(double d)
{
    return cbrt(d);
}
#endif

inline SHARE double Exp(double d)
{
    return exp(d);
}

inline SHARE double Exp10(double p)
{
    return pow(10.0, p);
}

inline SHARE double Hypot(double x, double y)
{
    return hypot(x, y);
}

inline SHARE double Sqr(double x)
{
    return x*x;
}

#endif
