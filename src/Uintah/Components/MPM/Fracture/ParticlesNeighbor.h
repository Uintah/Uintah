#ifndef __PARTICLESNEIGHBOR_H__
#define __PARTICLESNEIGHBOR_H__

#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Geometry/Point.h>

#include <vector>

class Matrix3;

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
class Lattice;
class LeastSquare;

class ParticlesNeighbor : public std::vector<particleIndex> {
public:

        ParticlesNeighbor();
        ParticlesNeighbor(const ParticleVariable<Point>& pX,
	                  const ParticleVariable<int>& pIsBroken,
			  const ParticleVariable<Vector>& pCrackSurfaceNormal,
			  const ParticleVariable<double>& pMicrocrackSize,
			  const ParticleVariable<double>& pMicrocrackPosition);

  const ParticleVariable<int>& getpIsBroken() const;
  
  void  buildIn(const IntVector& cellIndex,const Lattice& lattice);
  
  bool  visible(const Point& A,const Point& B) const;

  void  interpolateVector(LeastSquare& ls,
                          const particleIndex& pIdx,
                          const ParticleVariable<Vector>& pVector,
                          Vector& data, 
                          Matrix3& gradient) const;

  void  interpolatedouble(LeastSquare& ls,
                          const particleIndex& pIdx,
                          const ParticleVariable<double>& pdouble,
                          double& data,
                          Vector& gradient) const;

  void  interpolateInternalForce(LeastSquare& ls,
                          const particleIndex& pIdx,
                          const ParticleVariable<Matrix3>& pStress,
                          Vector& pInternalForce) const;

private:
  const ParticleVariable<Point>*  d_pX;
  const ParticleVariable<int>*    d_pIsBroken;
  const ParticleVariable<Vector>* d_pCrackSurfaceNormal;
  const ParticleVariable<double>* d_pMicrocrackSize;
  const ParticleVariable<double>* d_pMicrocrackPosition;
};

} //namespace MPM
} //namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__

// $Log$
// Revision 1.9  2000/09/12 16:52:11  tan
// Reorganized crack surface contact force algorithm.
//
// Revision 1.8  2000/09/11 00:15:00  tan
// Added calculations on random distributed microcracks in broken particles.
//
// Revision 1.7  2000/09/08 18:25:35  tan
// Added visibility calculation to fracture broken cell shape function
// interpolation.
//
// Revision 1.6  2000/07/06 16:59:24  tan
// Least square interpolation added for particle velocities and stresses
// updating.
//
// Revision 1.5  2000/07/06 06:23:08  tan
// Added Least Square interpolation of double (such as temperatures),
// vector (such as velocities) and stresses for particles in the
// self-contact cells.
//
// Revision 1.4  2000/06/23 21:56:30  tan
// Use vector instead of list for cells-neighbor and particles-neighbor.
//
// Revision 1.3  2000/06/06 01:58:14  tan
// Finished functions build particles neighbor for a given particle
// index.
//
// Revision 1.2  2000/06/05 22:30:02  tan
// Added interpolateVector and interpolatedouble for least-square approximation.
//
// Revision 1.1  2000/06/05 21:15:21  tan
// Added class ParticlesNeighbor to handle neighbor particles searching.
//
