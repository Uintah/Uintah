#include "Lattice.h"

#include "Cell.h"
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {
Lattice::
Lattice(const ParticleVariable<Point>& pX)
: Array3<Cell>( pX.getParticleSubset()->getPatch()->getCellLowIndex()
                  - IntVector(1,1,1),
                pX.getParticleSubset()->getPatch()->getCellHighIndex()
		  + IntVector(1,1,1) ),
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
} // End namespace Uintah
  

