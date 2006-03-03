
#include <Packages/Uintah/Dataflow/Modules/Operators/MMS/MMS1.h>

#include <math.h>

// Depending on speed issues and how large these functions become, it
// is possible that they should be moved to the .h files.

// !!!!WHAT IS A?  It should be renamed with a better name!!!!
// A is amplitude, take any math class.

double MMS::A_ = 1.0;
double MMS::viscosity_ = 2.0e-5;
double MMS::p_ref_ = 101325.0;

double
MMS1::pressure( double x_pos, double y_pos, double time )
{
  return p_ref_ - (0.25*A_*A_* 
                   ( cos(2.0*(x_pos-time))+cos(2.0*(y_pos-time)) )*
                   exp(-4.0*viscosity_*time));
}

double
MMS1::uVelocity( double x_pos, double y_pos, double time )
{
  return 1.0 - A_*cos(x_pos-time)*sin(y_pos-time)*exp(-2.0*viscosity_*time);
}

double
MMS1::vVelocity( double x_pos, double y_pos, double time )
{
  return 1.0 + A_*sin(x_pos-time)*cos(y_pos-time)*exp(-2.0*viscosity_*time);
}
  

