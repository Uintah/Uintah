/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

// 2 Unsigned Integers
inline SCICORESHARE unsigned int Min(unsigned int d1, unsigned int d2)
{
    return d1<d2?d1:d2;
}

inline SCICORESHARE unsigned int Max(unsigned int d1, unsigned int d2)
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

// 3 unsigned integers
inline SCICORESHARE unsigned int Min(unsigned int d1,
				     unsigned int d2,
				     unsigned int d3)
{
    unsigned int m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline SCICORESHARE unsigned int Mid(unsigned int a,
				     unsigned int b,
				     unsigned int c)
{
  return ((a > b) ? ((a < c) ? a : ((b > c) ? b : c)) : \
	            ((b < c) ? b : ((a > c) ? a : c)));
}

inline SCICORESHARE unsigned int Max(unsigned int d1,
				     unsigned int d2,
				     unsigned int d3)
{
    unsigned int m=d1>d2?d1:d2;
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
