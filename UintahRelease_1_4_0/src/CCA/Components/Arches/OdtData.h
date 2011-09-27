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


#ifndef Uintah_Component_Arches_OdtData_h
#define Uintah_Component_Arches_OdtData_h

#include <Core/Disclosure/TypeDescription.h>

namespace Uintah {
  struct odtData { 
  double x_x[10];   //x coordinates along x direction line
  double y_y[10];   //y coordinates along x direction line
  double z_z[10];   //z coordinates along x direction line
  
  double x_u[10];   //u velocity along x direction line
  double x_v[10];   //v velocity along x direction line
  double x_w[10];   //w velocity along x direction line
  double x_rho[10]; //density along x direction line 
  double x_T[10];   //temperature along x direction line
  double x_Phi[10]; //scalar  along x direction line
  
  double y_u[10];   //u velocity along y direction line
  double y_v[10];   //v velocity along y direction line
  double y_w[10];   //w velocity along y direction line
  double y_rho[10]; //density along y direction line 
  double y_T[10];   //temperature along y direction line
  double y_Phi[10]; //scalar  along y direction line
  
  double z_u[10];   //u velocity along z direction line
  double z_v[10];   //v velocity along z direction line
  double z_w[10];   //w velocity along z direction line
  double z_rho[10]; //density along z direction line 
  double z_T[10];   //temperature along z direction line
  double z_Phi[10]; //scalar  along z direction line
  };    


//________________________________________________________________________
  const TypeDescription* fun_getTypeDescription(odtData*);    

} // End namespace Uintah
  
namespace SCIRun {
  void swapbytes( Uintah::odtData& d);
}       

#endif
