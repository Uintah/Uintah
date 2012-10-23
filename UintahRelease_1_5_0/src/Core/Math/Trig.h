/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Math_Trig_h
#define Math_Trig_h 1

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double Pi=M_PI;

#ifdef _WIN32

inline double acosh(double x)
{
	return (x<1) ? log(-1.0) : log(x+sqrt(x*x-1));
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

inline double Atan2(double y, double x)
{
  return atan2(y, x);
}

inline double DtoR(double d)
{
    return d*(M_PI/180.);
}

inline double RtoD(double r)
{
    return r*(180./M_PI);
}

inline double ACosh(double x)
{
    return acosh(x);
}

/////////////////////////////////////////////////////
//
// Float version
//

inline float Sin(float d)
{
  return sinf(d);
}

inline float Cos(float d)
{
  return cosf(d);
}

inline float Tan(float d)
{
  return tanf(d);
}

inline float Atan(float d)
{
  return atanf(d);
}

inline float Atan2(float y, float x)
{
  return atan2f(y, x);
}

inline float Acos(float d)
{
  return acosf(d);
}

inline float DtoR(float d)
{
  return d*(float)(M_PI/180.0);
}

inline float RtoD(float r)
{
  return r*(float)(180.0/M_PI);
}

#endif
