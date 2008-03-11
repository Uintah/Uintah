
#include <Packages/Uintah/StandAlone/tools/compare_mms/SineMMS.h>

#include <math.h>
#include <iostream>

using std::cout;

//__________________________________
// Reference:  "A non-trival analytical solution to the 2d incompressible
//        Navier-Stokes equations" by Randy McDermott, August 12 2003

SineMMS::SineMMS(double A, double viscosity, double p_ref) {
	d_A=A;
	d_viscosity=viscosity;
	d_p_ref=p_ref;
}

SineMMS::~SineMMS() {
};

double
SineMMS::pressure( double x, double y, double z, double time )
{
  return d_p_ref - ( 0.25 * d_A * d_A * 
                   ( cos( 2.0*(x-time) ) + cos( 2.0*(y-time))) * exp( -4.0 * d_viscosity * time ) );
}

double
SineMMS::uVelocity( double x, double y, double z, double time )
{
  return 1- d_A*cos(x-time)*sin(y-time)*exp(-2.0*d_viscosity*time);
}
  
double
SineMMS::vVelocity( double x, double y, double z, double time )
{
  return 1+ d_A*sin(x-time)*cos(y-time)*exp(-2.0*d_viscosity*time);
}

double
SineMMS::wVelocity( double x, double y, double z, double time )
{
  return -9999999;
}

