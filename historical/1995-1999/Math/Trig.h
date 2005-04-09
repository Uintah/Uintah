
#ifndef Math_Trig_h
#define Math_Trig_h 1

#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

const double Pi=PI;

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
