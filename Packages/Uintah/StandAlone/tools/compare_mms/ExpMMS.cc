
#include <Packages/Uintah/StandAlone/tools/compare_mms/ExpMMS.h>

#include <math.h>

// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

// !!!!WHAT IS A?  It should be renamed with a better name!!!!

ExpMMS::ExpMMS(double A, double viscosity, double p_ref) {
    	d_A=A;
	d_viscosity=viscosity;
	d_p_ref=p_ref;
};
ExpMMS::~ExpMMS() {
};

double
ExpMMS::pressure( double x, double y, double z, double time )
{
  return d_p_ref - ( 0.25 * d_A * d_A * 
                   ( cos( 2.0*(x-time) ) + cos( 2.0*(y-time))) * exp( -4.0 * d_viscosity * time ) );
}

double
ExpMMS::uVelocity( double x, double y, double z, double time )
{
  return 1- d_A*cos(x-time)*sin(y-time)*exp(-2.0*d_viscosity*time);
}
  
double
ExpMMS::vVelocity( double x, double y, double z, double time )
{
  return 1- d_A*cos(x-time)*sin(y-time)*exp(-2.0*d_viscosity*time);
}

double
ExpMMS::wVelocity( double x, double y, double z, double time )
{
  return 1- d_A*cos(x-time)*sin(y-time)*exp(-2.0*d_viscosity*time);
}

