/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/Advection/SecondOrderBase.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Patch.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;
/*_____________________________________________________________________
Reference:  "W.B. VanderHeyden and B.A. Kashiwa, "Compatible Fluxes for
            van Leer Advection", JCP, 146, 1-28, 1998
_____________________________________________________________________*/
SecondOrderBase::SecondOrderBase()
{
}

SecondOrderBase::~SecondOrderBase()
{
  // Destructor
}


/*_____________________________________________________________________
 Function~ mass_massVertex_ratio
 Purpose   compute mass_CC/mass_vertex with gradient limited gradients 
_____________________________________________________________________*/
void
SecondOrderBase::mass_massVertex_ratio(const CCVariable<double>& mass_CC,
                                       const Patch* patch,
                                           const CCVariable<double>& mass_grad_x,
                                           const CCVariable<double>& mass_grad_y,
                                           const CCVariable<double>& mass_grad_z)
{
  Vector dx = patch->dCell();
  double dx_2 = dx.x()/2.0;  // distance from cell center to vertex
  double dy_2 = dx.y()/2.0;
  double dz_2 = dx.z()/2.0;
  
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {
    const IntVector& c = *iter;
    double xterm = mass_grad_x[c] * dx_2;
    double yterm = mass_grad_y[c] * dy_2;
    double zterm = mass_grad_z[c] * dz_2;
    double mass = mass_CC[c];
    
    vertex<double>& mass_ratio = d_mass_massVertex[c];
    mass_ratio.d_vrtx[0] = mass/(mass - xterm - yterm - zterm);
    mass_ratio.d_vrtx[1] = mass/(mass - xterm - yterm + zterm);
    mass_ratio.d_vrtx[2] = mass/(mass - xterm + yterm - zterm);     
    mass_ratio.d_vrtx[3] = mass/(mass - xterm + yterm + zterm);
    mass_ratio.d_vrtx[4] = mass/(mass + xterm - yterm - zterm);
    mass_ratio.d_vrtx[5] = mass/(mass + xterm - yterm + zterm);    
    mass_ratio.d_vrtx[6] = mass/(mass + xterm + yterm - zterm);
    mass_ratio.d_vrtx[7] = mass/(mass + xterm + yterm + zterm);
  }
}
