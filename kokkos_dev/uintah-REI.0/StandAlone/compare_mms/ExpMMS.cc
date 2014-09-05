
#include <Packages/Uintah/StandAlone/compare_mms/ExpMMS.h>

#include <math.h>

// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

// !!!!WHAT IS A?  It should be renamed with a better name!!!!

ExpMMS::ExpMMS() {
};
ExpMMS::~ExpMMS() {
};

double
ExpMMS::pressure( double x, double y, double z, double time )
{
  return p_ref_ - ( 0.25 * A_ * A_ * 
                   ( cos( 2.0*(x-time) ) + cos( 2.0*(y-time))) * exp( -4.0 * viscosity_ * time ) );
}

double
ExpMMS::uVelocity( double x, double y, double z, double time )
{
  return 1- A_*cos(x-time)*sin(y-time)*exp(-2.0*viscosity_*time);
}
  
double
ExpMMS::vVelocity( double x, double y, double z, double time )
{
  return 1- A_*cos(x-time)*sin(y-time)*exp(-2.0*viscosity_*time);
}

double
ExpMMS::wVelocity( double x, double y, double z, double time )
{
  return 1- A_*cos(x-time)*sin(y-time)*exp(-2.0*viscosity_*time);
}

