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
* 
* Heavily modified by David McAllister, 1998.
*/


#ifndef SCI_Math_MiscMath_h
#define SCI_Math_MiscMath_h 1

#include <math.h>
#include <Rendering/ZTex/Utils.h>

namespace SemotusVisum {
namespace Rendering {

#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif

inline double DtoR(double d)
{
    return d*M_PI/180.;
}

inline double RtoD(double r)
{
    return r*180./M_PI;
}

inline double Pow(double d, double p)
{
    return pow(d,p);
}

inline int Sqrt(int i)
{
    return (int)sqrt((double)i);
}

inline double Sqrt(double d)
{
    return sqrt(d);
}

inline double Cbrt(double d)
{
    return cbrt(d);
}

inline double Exp(double d)
{
    return exp(d);
}

inline double Exp10(double p)
{
    return pow(10.0, p);
}

inline double Hypot(double x, double y)
{
    return hypot(x, y);
}

inline double Sqr(double x)
{
    return x*x;
}

inline int Sqr(int x)
{
    return x*x;
}

inline double Cube(double x)
{
    return x*x*x;
}

inline int Cube(int x)
{
    return x*x*x;
}

// The sqrt of 2 pi.
#define SQRT2PI 2.506628274631000502415765284811045253006

// Compute the gaussian with sigma and mu at value x.
inline double Gaussian(double x, double sigma, double mu=0)
{
	return exp(-0.5 * Sqr(((x-mu)/sigma))) / (SQRT2PI * sigma);
}

// Symmetric gaussian centered at origin.
// No covariance matrix. Give it X and Y.
inline double Gaussian2(double x, double y, double sigma)
{
	return exp(-0.5 * (Sqr(x) + Sqr(y)) / Sqr(sigma)) / (SQRT2PI * sigma);
}

// Compute the gaussian with sigma and mu at value x.
// Useful when we don't want to take the sqrt.
inline double GaussianSq(double xSq, double sigma)
{
	return exp(-0.5 * xSq / Sqr(sigma)) / (SQRT2PI * sigma);
}

// Return a random number with a normal distribution.
inline double NRand(double sigma = 1.0)
{
#define ONE_OVER_SIGMA_EXP (1.0f / 0.7975f)

	double y;
	do
	{
		y = -log(DRand());
	}
	while(DRand() > exp(-Sqr(y-1)*0.5));

	if(LRand() & 0x1)
		return y * sigma * ONE_OVER_SIGMA_EXP;
	else
		return -y * sigma * ONE_OVER_SIGMA_EXP;
}

// Absolute value
inline double Abs(double d)
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
inline double Clamp(double min, double d, double max)
{
	return d<=min?min:d>=max?max:d;
}

inline int Clamp(int min, int i, int max)
{
	return i<min?min:i>max?max:i;
}

// 2 Integers
inline int Min(int d1, int d2)
{
    return d1<d2?d1:d2;
}

inline int Max(int d1, int d2)
{
    return d1>d2?d1:d2;
}

// 2 doubles
inline double Max(double d1, double d2)
{
  return d1>d2?d1:d2;
}

inline double Min(double d1, double d2)
{
  return d1<d2?d1:d2;
}

// 3 doubles
inline double Min(double d1, double d2, double d3)
{
    double m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline double Max(double d1, double d2, double d3)
{
    double m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 integers
inline int Min(int d1, int d2, int d3)
{
    int m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline int Max(int d1, int d2, int d3)
{
    int m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// Generate a step between min and max.
// return:   0 - if d<=min
//	         1 - if d>=max
//	         hermite curve if d>min && d<max
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

// Interpolation:
inline double Interpolate(double d1, double d2, double weight)
{
	return d2*weight+d1*(1.0-weight);
}

// Integer/double conversions
inline double Fraction(double d)
{
	if(d>0){
		return d-(int)d;
	} else {
		return d-(int)d+1;
	}
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
	if(d>=0){
		return (int)d;
	} else {
		return -(int)(-d);
	}
}

inline int Ceil(double d)
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


} // namespace Tools
} // namespace Remote

#endif /* SCI_Math_MiscMath_h */
