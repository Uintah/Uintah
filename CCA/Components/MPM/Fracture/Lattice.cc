#include "Lattice.h"

#include "Cell.h"
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {
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
    const Point& v = pX_patchOnly[*iter_patchOnly];
    for(ParticleSubset::iterator iter_patchAndGhost = pset_patchAndGhost->begin();
         iter_patchAndGhost != pset_patchAndGhost->end(); iter_patchAndGhost++)
    {
      const Point& p = pX_patchAndGhost[*iter_patchAndGhost];
      if( v.x() == p.x() && 
          v.y() == p.y() && 
          v.z() == p.z() )
      {
        particleIndexExchange[*iter_patchOnly] = *iter_patchAndGhost;
	break;
      }
    }
  }
}

} // End namespace Uintah
  

