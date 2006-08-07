
#include <Packages/Uintah/StandAlone/compare_mms/SineMMS.h>

#include <math.h>
#include <iostream>
using std::cout;
// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

// !!!!WHAT IS A?  It should be renamed with a better name!!!!


SineMMS::SineMMS(double A, double viscosity, double p_ref) {
	A_=A;
	viscosity_=viscosity;
	p_ref_=p_ref;
}

SineMMS::~SineMMS() {
};

double
SineMMS::pressure( double x, double y, double z, double time )
{
  return p_ref_ - ( 0.25 * A_ * A_ * 
                   ( cos( 2.0*(x-time) ) + cos( 2.0*(y-time))) * exp( -4.0 * viscosity_ * time ) );
}

double
SineMMS::uVelocity( double x, double y, double z, double time )
{
  return 1- A_*cos(x-time)*sin(y-time)*exp(-2.0*viscosity_*time);
}
  
double
SineMMS::vVelocity( double x, double y, double z, double time )
{
  return 1+ A_*sin(x-time)*cos(y-time)*exp(-2.0*viscosity_*time);
}

double
SineMMS::wVelocity( double x, double y, double z, double time )
{
  return 1+ A_*sin(x-time)*cos(y-time)*exp(-2.0*viscosity_*time);
}

