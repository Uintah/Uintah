
#ifndef Math_Trig_h
#define Math_Trig_h 1

#include <share/share.h>

#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

const double Pi=PI;

#ifdef _WIN32

inline SHARE double acosh(double x)
{
	return (x<1) ? log(-1) : log(x+sqrt(x*x-1));
}

#endif

inline SHARE double Cos(double d)
{
    return cos(d);
}

inline SHARE double Sin(double d)
{
    return sin(d);
}

inline SHARE double Asin(double d)
{
    return asin(d);
}

inline SHARE double Acos(double d)
{
    return acos(d);
}

inline SHARE double Tan(double d)
{
    return tan(d);
}

inline SHARE double Cot(double d)
{
    return 1./tan(d);
}

inline SHARE double Atan(double d)
{
    return atan(d);
}

inline SHARE double DtoR(double d)
{
    return d*PI/180.;
}

inline SHARE double RtoD(double r)
{
    return r*180./PI;
}

inline SHARE double ACosh(double x)
{
    return acosh(x);
}

#endif
