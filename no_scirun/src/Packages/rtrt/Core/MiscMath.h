/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#ifndef MISCMATH_H
#define MISCMATH_H 1

#include <cmath>

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
