#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Core/Geometry/Point.h>
#include "Cell.h"

namespace Uintah {

using namespace SCIRun;

class Lattice : public Array3<Cell> {
public:
        Lattice(const ParticleVariable<Point>& pX);

  bool              containCell(const IntVector& cellIndex) const;
  
  const Patch*                    getPatch() const;
  const ParticleVariable<Point>&  getpX() const;

private:
  const Patch*                   d_patch;
  const ParticleVariable<Point>& d_pX;
};

void fit(ParticleSubset* pset_patchOnly,
	 const ParticleVariable<Point>& pX_patchOnly,
         ParticleSubset* pset_patchAndGhost,
	 const ParticleVariable<Point>& pX_patchAndGhost,
	 std::vector<int>& particleIndexExchange);

} // End namespace Uintah

#endif //__LATTICE_H__

