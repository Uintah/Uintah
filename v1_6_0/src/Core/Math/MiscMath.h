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

} // End namespace SCIRun


#endif /* SCI_Math_MiscMath_h */
