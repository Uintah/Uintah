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
 *  MiscMath.h: Assorted mathematical functions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Math_MiscMath_h
#define SCI_Math_MiscMath_h 1

#include <Core/share/share.h>

namespace SCIRun {

// Absolute value
inline SCICORESHARE double Abs(double d)
{
  return d<0?-d:d;
}

inline SCICORESHARE int Abs(int i)
{
    return i<0?-i:i;
}

// Signs
inline SCICORESHARE int Sign(double d)
{
  return d<0?-1:1;
}

inline SCICORESHARE int Sign(int i)
{
  return i<0?-1:1;
}

// Clamp a number to a specific range
inline SCICORESHARE double Clamp(double d, double min, double max)
{
  return d<=min?min:d>=max?max:d;
}

inline SCICORESHARE int Clamp(int i, int min, int max)
{
  return i<min?min:i>max?max:i;
}

// Generate a step between min and max.
// return:   min - if d<=min
//	     max - if d>=max
//	     hermite curve if d>min && d<max
inline SCICORESHARE double SmoothStep(double d, double min, double max)
{
  double ret;
  if(d <= min){
    ret=0.0;
  } else if(d >= max){
    ret=1.0;
  } else {
    double dm=max-min;
    double t=(d-min)/dm;
    ret=t*t*(3.0-2.0*t);
  }
  return ret;
}

// Interpolation:
template <class T>
inline SCICORESHARE T Interpolate(T d1, T d2, double weight)
{
  return T(d2*weight+d1*(1.0-weight));
}




// Integer/double conversions
inline SCICORESHARE double Fraction(double d)
{
  if(d>0){
    return d-(int)d;
  } else {
    return d-(int)d+1;
  }
}

inline SCICORESHARE int RoundDown(double d)
{
  if(d>=0){
    return (int)d;
  } else {
    if(d==(int)d){
      return -(int)(-d);
    } else {
      return -(int)(-d)-1;
    }
  }
}

inline SCICORESHARE int RoundUp(double d)
{
    if(d>=0){
	if((d-(int)d) == 0)
	    return (int)d;
	else 
	    return (int)(d+1);
    } else {
	return (int)d;
    }
}

inline SCICORESHARE int Round(double d)
{
  return (int)(d+0.5);
}


inline SCICORESHARE int Floor(double d)
{
  if(d<0){
    int i=-(int)-d;
    if((double)i == d)
      return i;
    else
      return i-1;
  } else {
    return (int)d;
  }
}

inline SCICORESHARE int Ceil(double d)
{
  if(d<0){
    int i=-(int)-d;
    return i;
  } else {
    int i=(int)d;
    if((double)i == d)
      return i;
    else
      return i+1;
  }
}

inline SCICORESHARE int Tile(int tile, int tf)
{
  if(tf<0){
    // Tile in negative direction
      if(tile>tf && tile<=0)return 1;
      else return 0;
  } else if(tf>0){
    // Tile in positive direction
      if(tile<tf && tile>=0)return 1;
      else return 0;
  } else {
    return 1;
  }
}

SCICORESHARE double MakeReal(double value);

} // End namespace SCIRun


#endif /* SCI_Math_MiscMath_h */
