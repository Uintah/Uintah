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

#endif
