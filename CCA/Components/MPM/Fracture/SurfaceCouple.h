#ifndef __Uintah_SurfaceCouple__
#define __Uintah_SurfaceCouple__

#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Core/Geometry/Point.h>

#include <vector>

namespace Uintah {
using namespace SCIRun;

class Matrix3;
class Lattice;
class LeastSquare;

class SurfaceCouple {
public:

        SurfaceCouple();
	
  void  setup(particleIndex pIdxA,
              particleIndex pIdxB,
	      const Vector& normal);

  bool  extensible(
       particleIndex pIdx,
       const ParticleVariable<Point>& pX,
       const ParticleVariable<Vector>& pExtensionDirection,
       double volume,
       double& distanceToCrack) const;

private:
  particleIndex   d_pIdxA;
  particleIndex   d_pIdxB;
  Vector          d_normal;
};

} // End namespace Uintah

#endif //__Uintah_SurfaceCouple__
