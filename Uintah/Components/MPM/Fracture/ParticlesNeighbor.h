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

        ParticlesNeighbor(const ParticleVariable<Point>& pX);

  void  buildIncluding(const particleIndex& pIndex,
                       const Lattice& lattice);

  void  buildExcluding(const particleIndex& pIndex,
                       const Lattice& lattice);

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
  const ParticleVariable<Point>&  d_pX;
};

} //namespace MPM
} //namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__

// $Log$
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
