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

#include <Core/Math/share.h>
#include <cmath>
#include <cfloat>

namespace SCIRun {

// Absolute value
inline double Abs(double d)
{
  return d<0?-d:d;
}

inline float Abs(float d)
{
  return d<0?-d:d;
}

inline int Abs(int i)
{
    return i<0?-i:i;
}

// Signs
inline int Sign(double d)
{
  return d<0?-1:1;
}

inline int Sign(int i)
{
  return i<0?-1:1;
}

// Clamp a number to a specific range
inline double Clamp(double d, double min, double max)
{
  return d<=min?min:d>=max?max:d;
}

inline float Clamp(float d, float min, float max)
{
  return d<=min?min:d>=max?max:d;
}

inline int Clamp(int i, int min, int max)
{
  return i<min?min:i>max?max:i;
}

// Generate a step between min and max.
// return:   min - if d<=min
//           max - if d>=max
//           hermite curve if d>min && d<max
//
// Make sure fixes end up in both the float and double versions.
inline double SmoothStep(double d, double min, double max)
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

// Make sure fixes end up in both the float and double versions.
inline float SmoothStep(float d, float min, float max)
{
  float ret;
  if(d <= min){
    ret=0;
  } else if(d >= max){
    ret=1;
  } else {
    float dm=max-min;
    float t=(d-min)/dm;
    ret=t*t*(3-2*t);
  }
  return ret;
}

// Interpolation:
template <class T>
inline T Interpolate(T d1, T d2, double weight)
{
  return T(d2*weight+d1*(1.0-weight));
}
template <class T>
inline T Interpolate(T d1, T d2, float weight)
{
  return T(d2*weight+d1*(1-weight));
}




// Integer/double conversions


// The mantissa of a double is 52 bits, so in order to subtract out
// the whole component of the floating point number you need a 64 bit
// integer.  You can, of course, represent numbers larger than 2^64
// with a double, but numbers larger than 2^52 (or so) will not have
// the precision to contain a fractional component.  For such numbers
// the behavior of this function is undefined.

// This should compute (d - Floor(d)).  Examples of input/output:
//   0   ->   0
//   0.5 ->   0.5
//   0.3 ->   0.3
//   1   ->   0
//   1.6 ->   1.6
//   2   ->   0
//
//  -0.5 ->   0.5
//  -0.3 ->   0.7
//  -1   ->   0
//  -1.6 ->   0.4
//  -2   ->   0
inline double Fraction(double d)
{
  double frac = d-(long long)d;
  if (frac >= 0)
    return frac;
  else
    return frac+1;
}

inline float Fraction(float d)
{
  float frac = d-(int)d;
  if (frac >= 0)
    return frac;
  else
    return frac+1;
}

inline int RoundDown(double d)
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

inline int RoundUp(double d)
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

inline int Round(double d)
{
  return (int)(d+0.5);
}

inline int Floor(double d)
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

inline int Floor(float d)
{
  if(d<0){
    int i=-(int)-d;
    if((float)i == d)
      return i;
    else
      return i-1;
  } else {
    return (int)d;
  }
}

// this version is twice as fast for negative numbers
// (on an available Pentium 4 machine, only about 25%
// faster on Powerbook G4)
// than the more robust version above, but will only
//work for fabs(d) <= offset, use with caution
inline int Floor(double d, int offset)
{
   return (int)(d + offset) - offset;
}

inline int Floor(float d, int offset)
{
   return (int)(d + offset) - offset;
}

inline int Ceil(double d)
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

inline int Ceil(float d)
{
  if(d<0){
    int i=-(int)-d;
    return i;
  } else {
    int i=(int)d;
    if((float)i == d)
      return i;
    else
      return i+1;
  }
}

// using the same trick as above, this
// version of Ceil is a bit faster for
// the architectures it has been tested
// on (Pentium4, Powerbook G4)
inline int Ceil(double d, int offset)
{
   return (int)(d - offset) + offset;
}

inline int Ceil(float d, int offset)
{
   return (int)(d - offset) + offset;
}

inline int Tile(int tile, int tf)
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

double SCISHARE MakeReal(double value);


template <class T>
inline void SWAP(T& a, T& b) {
  T temp;
  temp = a;
  a = b;
  b = temp;
}

// Fast way to check for power of two
inline bool IsPowerOf2(unsigned int n)
{
  return (n & (n-1)) == 0;
}



// Returns a number Greater Than or Equal to dim
// that is an exact power of 2
// Used for determining what size of texture to
// allocate to store an image
inline unsigned int
Pow2(const unsigned int dim) {
  if (IsPowerOf2(dim)) return dim;
  unsigned int val = 4;
  while (val < dim) val = val << 1;;
  return val;
}

// Returns a number Less Than or Equal to dim
// that is an exact power of 2
inline unsigned int
LargestPowerOf2(const unsigned int dim) {
  if (IsPowerOf2(dim)) return dim;
  return Pow2(dim) >> 1;
}


// Returns the power of 2 of the next higher number thas is a power of 2
// Log2 of Pow2 funciton above
inline unsigned int
Log2(const unsigned int dim) {
  unsigned int log = 0;
  unsigned int val = 1;
  while (val < dim) { val = val << 1; log++; };
  return log;
}

// Takes the square root of "value" and tries to find two factors that
// are closest to that square root.
void findFactorsNearRoot(const int value, int& factor1, int& factor2);

//Computes the cubed root of a using Halley's algorithm.  The initial guess
//is set to the parameter provided, if no parameter is provided then the 
//result of the last call is used as the initial guess.  DBL_MIN is used
//as a sentinal to signal that no guess was provided.
double cubeRoot(double a, double guess=DBL_MIN);

} // End namespace SCIRun


#endif /* SCI_Math_MiscMath_h */
