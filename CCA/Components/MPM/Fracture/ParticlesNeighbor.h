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

  bool computeEnergyReleaseRate(
        particleIndex tipIndex,
        const Vector& nx,
	Vector& ny,
	double stress,
	const ParticleVariable<Point>& pX,
	const ParticleVariable<double>& pVolume,
	double& G) const;

private:

};

} // End namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__
