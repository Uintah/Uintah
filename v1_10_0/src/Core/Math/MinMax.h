/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

// 2 Long Integers
inline SCICORESHARE long Min(long d1, long d2)
{
    return d1<d2?d1:d2;
}

inline SCICORESHARE long Max(long d1, long d2)
{
    return d1>d2?d1:d2;
}

// 2 floats
inline SCICORESHARE float Max(float d1, float d2)
{
  return d1>d2?d1:d2;
}

inline SCICORESHARE float Min(float d1, float d2)
{
  return d1<d2?d1:d2;
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

inline SCICORESHARE double Mid(double a, double b, double c)
{
  return ((a > b) ? ((a < c) ? a : ((b > c) ? b : c)) : \
	            ((b < c) ? b : ((a > c) ? a : c)));
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

inline SCICORESHARE int Mid(int a, int b, int c)
{
  return ((a > b) ? ((a < c) ? a : ((b > c) ? b : c)) : \
	            ((b < c) ? b : ((a > c) ? a : c)));
}

inline SCICORESHARE int Max(int d1, int d2, int d3)
{
    int m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 Long Integers
inline SCICORESHARE long  Min(long  d1, long  d2, long  d3)
{
    long  m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline SCICORESHARE long Mid(long a, long b, long c)
{
  return ((a > b) ? ((a < c) ? a : ((b > c) ? b : c)) : \
	            ((b < c) ? b : ((a > c) ? a : c)));
}


inline SCICORESHARE long  Max(long  d1, long  d2, long  d3)
{
    long  m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

} // End namespace SCIRun


#endif /* SCI_Math_MinMax_h */
