#ifndef __Uintah_IndexExchange__
#define __Uintah_IndexExchange__

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Core/Geometry/Point.h>
#include "Cell.h"

namespace Uintah {

using namespace SCIRun;
class ParticlesNeighbor;

class IndexExchange {
public:

  IndexExchange(ParticleSubset* pset_patchOnly,
                const ParticleVariable<Point>& pX_patchOnly,
		ParticleSubset* pset_patchAndGhost,
		const ParticleVariable<Point>& pX_patchAndGhost);

  int getPatchOnlyIndex(particleIndex pIdx_pg) const;
  int getPatchAndGhostIndex(particleIndex pIdx_p) const;
  	 
private:
  vector<int>  pIdxs_p;
  vector<int>  pIdxs_pg;
};

} // End namespace Uintah

#endif //__LATTICE_H__

