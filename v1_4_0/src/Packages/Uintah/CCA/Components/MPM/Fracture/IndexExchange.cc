#include "IndexExchange.h"

#include "Cell.h"
#include "ParticlesNeighbor.h"

#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {

IndexExchange::IndexExchange(ParticleSubset* pset_patchOnly,
                const ParticleVariable<Point>& pX_patchOnly,
		ParticleSubset* pset_patchAndGhost,
		const ParticleVariable<Point>& pX_patchAndGhost)
: pIdxs_p(pset_patchOnly->numParticles()),
  pIdxs_pg(pset_patchAndGhost->numParticles(),-1)
{
  Vector d = pset_patchOnly->getPatch()->dCell()/10;
  
  for(ParticleSubset::iterator iter_patchOnly = pset_patchOnly->begin();
       iter_patchOnly != pset_patchOnly->end(); iter_patchOnly++)
  {
    const Point& v = pX_patchOnly[*iter_patchOnly];
    for(ParticleSubset::iterator iter_patchAndGhost = pset_patchAndGhost->begin();
         iter_patchAndGhost != pset_patchAndGhost->end(); iter_patchAndGhost++)
    {
      const Point& p = pX_patchAndGhost[*iter_patchAndGhost];
      if( fabs(v.x()-p.x()) < d.x() && 
          fabs(v.y()-p.y()) < d.y() && 
          fabs(v.z()-p.z()) < d.z() )
      {
        pIdxs_p[*iter_patchOnly] = *iter_patchAndGhost;
        pIdxs_pg[*iter_patchAndGhost] = *iter_patchOnly;
	break;
      }
    }
  }
}

int IndexExchange::getPatchOnlyIndex(particleIndex pIdx_pg) const
{
  return pIdxs_pg[pIdx_pg];
}

int IndexExchange::getPatchAndGhostIndex(particleIndex pIdx_p) const
{
  return pIdxs_p[pIdx_p];
}

} // End namespace Uintah
  

