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


#ifndef Math_Trig_h
#define Math_Trig_h 1

#include <Core/share/share.h>

#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

const double Pi=PI;

#ifdef _WIN32

inline SCICORESHARE double acosh(double x)
{
	return (x<1) ? log(-1) : log(x+sqrt(x*x-1));
}

#endif

inline SCICORESHARE double Cos(double d)
{
    return cos(d);
}

inline SCICORESHARE double Sin(double d)
{
    return sin(d);
}

inline SCICORESHARE double Asin(double d)
{
    return asin(d);
}

inline SCICORESHARE double Acos(double d)
{
    return acos(d);
}

inline SCICORESHARE double Tan(double d)
{
    return tan(d);
}

inline SCICORESHARE double Cot(double d)
{
    return 1./tan(d);
}

inline SCICORESHARE double Atan(double d)
{
    return atan(d);
}

inline SCICORESHARE double DtoR(double d)
{
    return d*PI/180.;
}

inline SCICORESHARE double RtoD(double r)
{
    return r*180./PI;
}

inline SCICORESHARE double ACosh(double x)
{
    return acosh(x);
}

#endif
