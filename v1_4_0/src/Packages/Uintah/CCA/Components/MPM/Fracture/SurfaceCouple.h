#ifndef __Uintah_SurfaceCouple__
#define __Uintah_SurfaceCouple__

#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Core/Geometry/Point.h>

#include "Lattice.h"

#include <vector>

namespace Uintah {
using namespace SCIRun;

class SurfaceCouple {
public:

        SurfaceCouple() {};
	SurfaceCouple(particleIndex pIdxA,
                      particleIndex pIdxB,
	              const Vector& normal) :
	  d_pIdxA(pIdxA),d_pIdxB(pIdxB),d_normal(normal) {}
	
  void  setup(particleIndex pIdxA,
              particleIndex pIdxB,
	      const Vector& normal);

  particleIndex getIdxA() const;
  particleIndex getIdxB() const;
  
  const Vector& getNormal() const
    {
      return d_normal;
    }
    
  Point crackTip(const ParticleVariable<Point>& pX) const;
  void  tipMatrix(const ParticleVariable<Matrix3>& pMatrix,
                  Matrix3& matrix) const;
  void  tipVector(const ParticleVariable<Vector>& pVector,
                  Vector& vec) const;
  
  bool  extensible(
       particleIndex pIdx,
       const ParticleVariable<Point>& pX,
       const ParticleVariable<Vector>& pExtensionDirection,
       const ParticleVariable<Vector>& pCrackNormal,
       double volume,
       double& distanceToCrack) const;

  bool computeCrackClosureIntegralAndCrackNormalFromForce(
	const Vector& nxx,
	const Lattice& lattice,
	const ParticleVariable<Matrix3>& pStress,
	const ParticleVariable<Vector>& pDisplacement,
	const ParticleVariable<double>& pVolume,
	const ParticleVariable<int>& pIsBroken,
	double toughness,
	double& GI,double& GII,double& GIII,Vector& N ) const;

  bool computeCrackClosureIntegralAndCrackNormalFromEnergyReleaseRate(
	const Vector& nxx,
	const Lattice& lattice,
	const ParticleVariable<Matrix3>& pStress,
	const ParticleVariable<Vector>& pDisplacement,
	const ParticleVariable<double>& pVolume,
	const ParticleVariable<int>& pIsBroken,
	double toughness,
	double& GI,double& GII,double& GIII,Vector& N ) const;

private:
  particleIndex   d_pIdxA;
  particleIndex   d_pIdxB;
  Vector          d_normal;
};

} // End namespace Uintah

#endif //__Uintah_SurfaceCouple__
