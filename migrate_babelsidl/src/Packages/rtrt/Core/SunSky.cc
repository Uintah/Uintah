//==========================================================================
//
// Filename: SunSky.cc
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
#include <Packages/rtrt/Core/SunSky.h>
#endif

#include <Core/Math/MiscMath.h>

using namespace rtrt;

SunSky::SunSky(
		    double		lat,
		    double 	longi,
		    int 		sm,        		// standardMeridian
		    int 		jd,        		// julianDay
		    double		tOfDay,    		// timeOfDay
		    double 	turb 			// turbidity
)
{
	latitude = lat;
	longitude = longi;
	julianDay = jd;
	timeOfDay = tOfDay;
	standardMeridian = int(sm * 15.0);   // sm is actually timezone number (east to west, zero based...)
	turbidity = turb;

	InitSunThetaPhi();
	toSun = Vector(cos(phiS) * sin(thetaS), sin(phiS) * sin(thetaS), cos(thetaS));

	// Units: W.cm^-2.Sr^-1.
	sunSolidAngle = 0.25 * M_PI * 1.39 * 1.39 / (150 * 150);   // = 6.7443e-05

}

/**********************************************************
// South = x,  East = y, up = z
// All times in decimal form (6.25 = 6:15 AM)
// All angles in Radians
// ********************************************************/

void 
SunSky::GetSunThetaPhi( double& stheta, double& sphi ) const
{
	sphi = phiS;
	stheta = thetaS;
}

#define DegsToRads(x) x/180.*M_PI

/**********************************************************
// South = x,  East = y, up = z
// All times in decimal form (6.25 = 6:15 AM)
// All angles in Radians
// ********************************************************/

void 
SunSky::InitSunThetaPhi( void )
{
    double solarTime = timeOfDay +
	(0.170 * sin(4 * M_PI * (julianDay - 80) / 373) - 0.129 * sin(2 * M_PI* (julianDay - 8) / 355)) +
	(standardMeridian - longitude) / 15.0;
    
    double solarDeclination = (0.4093 * sin(2 * M_PI * (julianDay - 81) / 368));
    
    double solarAltitude = asin(sin(DegsToRads(latitude)) * sin(solarDeclination) -
				cos(DegsToRads(latitude)) * cos(solarDeclination) * cos(M_PI * solarTime / 12));
    
    double  opp, adj;
    opp = -cos(solarDeclination) * sin(M_PI * solarTime / 12);
    adj = -(cos(DegsToRads(latitude)) * sin(solarDeclination) +
	    sin(DegsToRads(latitude)) * cos(solarDeclination) * cos(M_PI * solarTime / 12));
    double solarAzimuth = atan2(opp, adj);
    
    phiS = -solarAzimuth;
    thetaS = M_PI / 2.0 - solarAltitude;
}

Vector 
SunSky::GetSunPosition( void ) const
{
    return toSun;
}

double 
SunSky::GetSunSolidAngle( void ) const
{
    return sunSolidAngle;
}


//// Copy Constructor
SunSky::SunSky( const SunSky & /*s*/ )
{
    // NOT IMPLEMENTED YET
}

//// Assignment
SunSky& 
SunSky::operator=( const SunSky& /*s*/ )
{
    // NOT IMPLEMENTED YET
    return( *this );
}
