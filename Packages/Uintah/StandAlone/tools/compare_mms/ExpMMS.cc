
#include <Packages/Uintah/StandAlone/tools/compare_mms/ExpMMS.h>

#include <cmath>

// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

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
  return -999999;
}

double
ExpMMS::uVelocity( double x, double y, double z, double time )
{
  return -999999;
}
  
double
ExpMMS::vVelocity( double x, double y, double z, double time )
{
  return -99999;
}

double
ExpMMS::wVelocity( double x, double y, double z, double time )
{
  return -99999;
}

