
#include <Packages/Uintah/StandAlone/compare_mms/MMS1.h>

#include <math.h>

// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

// !!!!WHAT IS A?  It should be renamed with a better name!!!!

double MMS::A_ = 1.0;
double MMS::viscosity_ = 2.0e-5;
double MMS::p_ref_ = 101325.0;

double
MMS1::pressure( int x, int y, double time )
{
  return p_ref_ - ( 0.25 * A_ * A_ * 
                   ( cos( 2.0*(x-time) ) + cos( 2.0*(y-time))) * exp( -4.0 * viscosity_ * time ) );
}

double
MMS1::uVelocity( int x, int y, double time )
{
  return 1- A_*cos(x-time)*sin(y-time)*exp(-2.0*viscosity_*time);
}
  

