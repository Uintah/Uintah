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

#include <Core/share/share.h>
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

inline SCICORESHARE double ipow(double x, int p)
{
  double result=1;
  while(p){
    if(p&1)
      result*=x;
    x*=x;
      p>>=1;
  }
  return result;
}
#endif
