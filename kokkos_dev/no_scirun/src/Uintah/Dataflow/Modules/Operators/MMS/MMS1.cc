/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



#include <Uintah/Dataflow/Modules/Operators/MMS/MMS1.h>

#include <cmath>

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
  

