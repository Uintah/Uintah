
/*
 *  MinMax.h: Various implementations of Min and Max
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Math_MinMax_h
#define SCI_Math_MinMax_h 1

#include <Core/share/share.h>

namespace SCIRun {

// 2 Integers
inline SCICORESHARE int Min(int d1, int d2)
{
    return d1<d2?d1:d2;
}

inline SCICORESHARE int Max(int d1, int d2)
{
    return d1>d2?d1:d2;
}

// 2 Integers
inline SCICORESHARE long Min(long d1, long d2)
{
    return d1<d2?d1:d2;
}

inline SCICORESHARE long Max(long d1, long d2)
{
    return d1>d2?d1:d2;
}

// 2 doubles
inline SCICORESHARE double Max(double d1, double d2)
{
  return d1>d2?d1:d2;
}

inline SCICORESHARE double Min(double d1, double d2)
{
  return d1<d2?d1:d2;
}

// 3 doubles
inline SCICORESHARE double Min(double d1, double d2, double d3)
{
    double m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline SCICORESHARE double Max(double d1, double d2, double d3)
{
    double m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 integers
inline SCICORESHARE int Min(int d1, int d2, int d3)
{
    int m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline SCICORESHARE int Max(int d1, int d2, int d3)
{
    int m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 integers
inline SCICORESHARE long  Min(long  d1, long  d2, long  d3)
{
    long  m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline SCICORESHARE long  Max(long  d1, long  d2, long  d3)
{
    long  m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

} // End namespace SCIRun


#endif /* SCI_Math_MinMax_h */
