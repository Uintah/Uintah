
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

#include <SCICore/share/share.h>

#include <math.h>

inline SCICORESHARE double Pow(double d, double p)
{
    return pow(d,p);
}

inline SCICORESHARE int Sqrt(int i)
{
    return (int)sqrt((double)i);
}

inline SCICORESHARE double Sqrt(double d)
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

inline SCICORESHARE double Exp(double d)
{
    return exp(d);
}

inline SCICORESHARE double Exp10(double p)
{
    return pow(10.0, p);
}

inline SCICORESHARE double Hypot(double x, double y)
{
    return hypot(x, y);
}

inline SCICORESHARE double Sqr(double x)
{
    return x*x;
}

#endif
