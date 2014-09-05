#ifndef __PARTICLESNEIGHBOR_H__
#define __PARTICLESNEIGHBOR_H__

#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Core/Geometry/Point.h>

#include <vector>

namespace Uintah {
using namespace SCIRun;

class Matrix3;
class Lattice;
class LeastSquare;

class ParticlesNeighbor : public std::vector<particleIndex> {
public:

        ParticlesNeighbor();

  void  buildIn(const IntVector& cellIndex,const Lattice& lattice);
  
  bool  visible(particleIndex idx,
                const Point& B,
		const ParticleVariable<Point>& pX,
		const ParticleVariable<int>& pIsBroken,
		const ParticleVariable<Vector>& pCrackSurfaceNormal1,
		const ParticleVariable<Vector>& pCrackSurfaceNormal2,
		const ParticleVariable<Vector>& pCrackSurfaceNormal3,
		const ParticleVariable<double>& pVolume) const;

  bool  visible(particleIndex idxA,
                particleIndex idxB,
	        const ParticleVariable<Point>& pX,
	        const ParticleVariable<int>& pIsBroken,
	        const ParticleVariable<Vector>& pCrackSurfaceNormal1,
	        const ParticleVariable<Vector>& pCrackSurfaceNormal2,
	        const ParticleVariable<Vector>& pCrackSurfaceNormal3,
	        const ParticleVariable<double>& pVolume) const;

  bool visible(const Point& A,
               const Point& B,
	       const Point& O,const Vector& N,double size2) const;

  double computeEnergyReleaseRate(
        particleIndex tipIndex,
        const Matrix3& stress,
	const ParticleVariable<Point>& pX,
	const ParticleVariable<double>& pVolume,
	const ParticleVariable<int>& pIsBroken,
	double& sigmaN,
	Vector& Ny,
	Vector& Nx) const;

  double computeCrackClosureIntegral(
        const Point& pTip,
	double R,
	const Vector& nx,
	Vector ny,
        const Matrix3& stress,
	const ParticleVariable<Point>& pX,
	const ParticleVariable<Vector>& pVelocity,
	const ParticleVariable<double>& pVolume,
	double delT ) const;

private:

};

} // End namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__
