
#ifndef MISCMATH_H
#define MISCMATH_H 1

#include <math.h>

//namespace rtrt {

//
///< Degrees to radians conversion...
//
inline double DegsToRads( double degrees )      
{ 
    return(degrees * ( M_PI / 180.0 ) ); 
}

///< Radians to degrees conversion...
inline double RadsToDegs( double rads ) 
{ 
    return(rads * ( 180.0 / M_PI ) ); 
}

inline double Abs(double x) {
    return x<0?-x:x;
}

inline double Clamp(double d, double min, double max)
{
        if(d <= min)return min;
        if(d >= max)return max;
        return d;
}

inline double Interpolate(double d1, double d2, double weight)
{
	return d1*weight+d2*(1.0-weight);
}

inline double SmoothStep(double d, double min, double max)
{
        if(d <= min)return 0.0;
        if(d >= max)return 1.0;
        double dm=max-min;
        double t=(d-min)/dm;
        return t*t*(3.0-2.0*t);
}

//} // end namespace rtrt

#endif
