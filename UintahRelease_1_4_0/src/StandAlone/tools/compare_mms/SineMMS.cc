/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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



#include <StandAlone/tools/compare_mms/SineMMS.h>

#include <cmath>
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

