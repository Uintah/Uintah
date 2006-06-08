#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderBase.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
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
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  Vector dx = patch->dCell();
  double dx_2 = dx.x()/2.0;  // distance from cell center to vertex
  double dy_2 = dx.y()/2.0;
  double dz_2 = dx.z()/2.0;
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
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
