
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

#include <share/share.h>

namespace SCICore {
namespace Math {

// Absolute value
inline SHARE double Abs(double d)
{
  return d<0?-d:d;
}

inline SHARE int Abs(int i)
{
    return i<0?-i:i;
}

// Signs
inline SHARE int Sign(double d)
{
  return d<0?-1:1;
}

inline SHARE int Sign(int i)
{
  return i<0?-1:1;
}

// Clamp a number to a specific range
inline SHARE double Clamp(double d, double min, double max)
{
  return d<=min?min:d>=max?max:d;
}

inline SHARE int Clamp(int i, int min, int max)
{
  return i<min?min:i>max?max:i;
}

// Generate a step between min and max.
// return:   min - if d<=min
//	     max - if d>=max
//	     hermite curve if d>min && d<max
inline SHARE double SmoothStep(double d, double min, double max)
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
inline SHARE double Interpolate(double d1, double d2, double weight)
{
  return d2*weight+d1*(1.0-weight);
}

// Integer/double conversions
inline SHARE double Fraction(double d)
{
  if(d>0){
    return d-(int)d;
  } else {
    return d-(int)d+1;
  }
}

inline SHARE int RoundDown(double d)
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

inline SHARE int RoundUp(double d)
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

inline SHARE int Round(double d)
{
  return (int)(d+0.5);
}


inline SHARE int Floor(double d)
{
  if(d>=0){
    return (int)d;
  } else {
    return -(int)(-d);
  }
}

inline SHARE int Ceil(double d)
{
  if(d==(int)d){
    return (int)d;
  } else {
    if(d>0){
      return 1+(int)d;
    } else {
      return -(1+(int)(-d));
    }
  }
}

inline SHARE int Tile(int tile, int tf)
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

} // End namespace Math
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:03  mcq
// Initial commit
//
// Revision 1.4  1999/07/01 16:44:23  moulding
// added SHARE to enable win32 shared libraries (dll's)
//
// Revision 1.3  1999/05/06 19:56:19  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:24  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//

#endif /* SCI_Math_MiscMath_h */
