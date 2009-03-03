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


//==========================================================================
//
// Filename: SunSky.h
//
//
//
//
//
//
//
// Description: Just some basic information about sun
//
//==========================================================================

#ifndef __SUNSKY_H__
#define __SUNSKY_H__

#if defined __cplusplus

#include <Core/Geometry/Vector.h>

namespace rtrt {

using SCIRun::Vector;

class SunSky
{

public:
    // GROUP: Constructors and assignment
    //// Default Constructor
    //    SunSky();
    //// Constructs an SunSky based on
    // [in] lat Latitude (0-360)
    // [in] long Longitude (-90,90) south to north
    // [in] sm  Standard Meridian
    // [in] jd  Julian Day (1-365)
    // [in] tod Time Of Day (0.0,23.99) 14.25 = 2:15PM
    // [in] turb  Turbidity (1.0,30+) 2-6 are most useful for clear days.
    SunSky(
                double          lat,
                double          longi,
		int             sm,
                int             jd,
                double          tod,
                double          turb
    );
    // GROUP: Members
    ////
    // South = x,  East = y, up = z
    Vector      GetSunPosition( void ) const;
    ////
    double      GetSunSolidAngle( void ) const;

    //// [out] theta  Sun angle down from straight above
    //   [out] phi    Sun angle anticlockwise from South
    void  GetSunThetaPhi( double& theta, double& phi ) const;

private:
    //// Copy Constructor
                            SunSky(const SunSky &);
    //// Assignment
                            SunSky &operator=(const SunSky &);

    //// Compute the sun's position based on IES Sunlight Publication ????
    void                InitSunThetaPhi();

    //Group: Data
    double     latitude;
    double     longitude;
    int        julianDay;
    double     timeOfDay;
    int        standardMeridian;
    double     turbidity;

    //// Sun Position Vector
    Vector              toSun;
    //// Sun Position
    double             thetaS, phiS;
    //// Solid angle of the sun when seen from earth
    double     sunSolidAngle;

};


} // end namespace rtrt

#endif
#endif
