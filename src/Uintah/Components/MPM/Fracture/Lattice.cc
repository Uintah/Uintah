#include "Lattice.h"

#include "Cell.h"
#include <Uintah/Grid/Patch.h>

namespace Uintah {
namespace MPM {

Lattice::
Lattice(const ParticleVariable<Point>& pX)
: Array3<Cell>( pX.getParticleSubset()->getPatch()->getCellLowIndex()
                  - IntVector(2,2,2),
                pX.getParticleSubset()->getPatch()->getCellHighIndex()
		  + IntVector(2,2,2) ),
    d_patch( pX.getParticleSubset()->getPatch() ),
    d_pX(pX)
{
  ParticleSubset* pset = d_pX.getParticleSubset();

  for(ParticleSubset::iterator part_iter = pset->begin();
      part_iter != pset->end(); part_iter++)
  {
    (*this)[ d_patch->getLevel()->getCellIndex(pX[*part_iter]) ].insert(*part_iter);
  }
}

bool Lattice::containCell(const IntVector& cellIndex) const
{
  if( cellIndex.x() >= getLowIndex().x() &&
      cellIndex.y() >= getLowIndex().y() &&
      cellIndex.z() >= getLowIndex().z() &&
      cellIndex.x() < getHighIndex().x() &&
      cellIndex.y() < getHighIndex().y() &&
      cellIndex.z() < getHighIndex().z() ) return true;
  else return false;
}

const Patch* Lattice::getPatch() const
{
  return d_patch;
}

const ParticleVariable<Point>& Lattice::getpX() const
{
  return d_pX;
}

void fit(ParticleSubset* pset_patchOnly,
	 const ParticleVariable<Point>& pX_patchOnly,
         ParticleSubset* pset_patchAndGhost,
	 const ParticleVariable<Point>& pX_patchAndGhost,
	 vector<int>& particleIndexExchange)
{
  for(ParticleSubset::iterator iter_patchOnly = pset_patchOnly->begin();
       iter_patchOnly != pset_patchOnly->end(); iter_patchOnly++)
  {
    for(ParticleSubset::iterator iter_patchAndGhost = pset_patchAndGhost->begin();
         iter_patchAndGhost != pset_patchAndGhost->end(); iter_patchAndGhost++)
    {
      if( pX_patchOnly[*iter_patchOnly] == 
          pX_patchAndGhost[*iter_patchAndGhost] )
      {
        particleIndexExchange[*iter_patchOnly] = *iter_patchAndGhost;
	break;
      }
    }
  }
}
  
} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.12  2000/12/30 05:08:10  tan
// Fixed a problem concerning patch and ghost in fracture computations.
//
// Revision 1.11  2000/09/25 20:23:21  sparker
// Quiet g++ warnings
//
// Revision 1.10  2000/09/05 19:38:42  tan
// Fracture starts to run in Uintah/MPM!
//
// Revision 1.9  2000/09/05 06:34:27  tan
// Introduced BrokenCellShapeFunction for SerialMPM::interpolateParticlesToGrid
// where farcture is involved.
//
// Revision 1.8  2000/06/20 02:29:22  tan
// Modification to make getCellIndex work.
//
// Revision 1.7  2000/06/15 21:57:09  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.6  2000/06/08 17:58:27  tan
// A small change
//
// Revision 1.5  2000/06/06 21:04:47  bigler
// Added const to Lattice members to get it to compile
//
// Revision 1.4  2000/06/05 23:57:51  tan
// Added conainCell().
//
// Revision 1.3  2000/06/05 22:32:38  tan
// Added function to find neighbor for a given particle index.
//
// Revision 1.2  2000/06/05 19:49:01  tan
// Added d_lattice which is a Array3 data of cells in a given patch.
//
// Revision 1.1  2000/06/05 17:22:05  tan
// Lattice class will be designed to make it easier to handle the grid/particle
// relationship in a given patch and a given velocity field.
//
