#ifndef __Uintah_BoundaryBand__
#define __Uintah_BoundaryBand__

#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Core/Geometry/Point.h>

#include <vector>

namespace Uintah {
using namespace SCIRun;

class Matrix3;
class Lattice;
class LeastSquare;

class BoundaryBand {
public:

        BoundaryBand();
	
  void  setup(particleIndex pIdx,
	      const ParticleVariable<Vector>& pCrackNormal,
	      const ParticleVariable<int>& pIsBroken,
	      const ParticleVariable<double>& pVolume,
	      const Lattice& lattice,
	      double range);

  void  setup(const Point& p,
	      const ParticleVariable<Vector>& pCrackNormal,
	      const ParticleVariable<int>& pIsBroken,
	      const ParticleVariable<double>& pVolume,
	      const Lattice& lattice,
	      double range);

  int   inside(particleIndex pIdx) const;
  int   inside(const Point& p) const;

  int   numCracks() const;
  
private:
  std::vector<particleIndex>      d_pIndexs;
  const ParticleVariable<Vector>* d_pCrackNormal;
  const ParticleVariable<Point>*  d_pX;
  const ParticleVariable<double>* d_pVolume;
  const ParticleVariable<int>*    d_pIsBroken;
  particleIndex                   d_idx;
};

} // End namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__
