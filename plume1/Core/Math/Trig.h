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



#ifndef Math_Trig_h
#define Math_Trig_h 1

#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

const double Pi=PI;

#ifdef _WIN32

inline double acosh(double x)
{
	return (x<1) ? log(-1) : log(x+sqrt(x*x-1));
}

#endif

inline double Cos(double d)
{
    return cos(d);
}

inline double Sin(double d)
{
    return sin(d);
}

inline double Asin(double d)
{
    return asin(d);
}

inline double Acos(double d)
{
    return acos(d);
}

inline double Tan(double d)
{
    return tan(d);
}

inline double Cot(double d)
{
    return 1./tan(d);
}

inline double Atan(double d)
{
    return atan(d);
}

inline double DtoR(double d)
{
    return d*PI/180.;
}

inline double RtoD(double r)
{
    return r*180./PI;
}

inline double ACosh(double x)
{
    return acosh(x);
}

#endif
