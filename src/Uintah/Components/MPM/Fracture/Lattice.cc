#include "Lattice.h"

#include "Cell.h"
#include <Uintah/Grid/Patch.h>

namespace Uintah {
namespace MPM {

Lattice::
Lattice(const Patch* patch,const ParticleVariable<Point>& pX)
: d_lattice(patch->getLowGhostCellIndex(),patch->getHighGhostCellIndex())
{
  
}
  
} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.2  2000/06/05 19:49:01  tan
// Added d_lattice which is a Array3 data of cells in a given patch.
//
// Revision 1.1  2000/06/05 17:22:05  tan
// Lattice class will be designed to make it easier to handle the grid/particle
// relationship in a given patch and a given velocity field.
//
