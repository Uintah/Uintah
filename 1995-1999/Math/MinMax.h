
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

// 2 Integers
inline int Min(int d1, int d2)
{
    return d1<d2?d1:d2;
}

inline int Max(int d1, int d2)
{
    return d1>d2?d1:d2;
}

// 2 doubles
inline double Max(double d1, double d2)
{
  return d1>d2?d1:d2;
}

inline double Min(double d1, double d2)
{
  return d1<d2?d1:d2;
}

// 3 doubles
inline double Min(double d1, double d2, double d3)
{
    double m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline double Max(double d1, double d2, double d3)
{
    double m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 integers
inline int Min(int d1, int d2, int d3)
{
    int m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline int Max(int d1, int d2, int d3)
{
    int m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

#endif /* SCI_Math_MinMax_h */
