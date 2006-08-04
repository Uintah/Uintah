
#include <Packages/Uintah/StandAlone/compare_mms/LinearMMS.h>

#include <math.h>

// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

// !!!!WHAT IS A?  It should be renamed with a better name!!!!

LinearMMS::LinearMMS(double cu, double cv, double cw, double cp, double p_ref) {
	c_u=cu;
	c_v=cv;
	c_w=cw;
	c_p=cp;
	p_ref_=p_ref;
};
LinearMMS::~LinearMMS() {
};

double
LinearMMS::pressure( double x, double y, double z, double time )
{
  return p_ref_ + c_p*( x + y + z) + time;
}

double
LinearMMS::uVelocity( double x, double y, double z, double time )
{
  return c_u*x + time;
}
  
double
LinearMMS::vVelocity( double x, double y, double z, double time )
{
  return c_v*y + time;
}

double
LinearMMS::wVelocity( double x, double y, double z, double time )
{
  return c_w*z + time;
}

