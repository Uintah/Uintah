#ifndef __PARTICLESNEIGHBOR_H__
#define __PARTICLESNEIGHBOR_H__

#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Geometry/Point.h>

#include <vector>

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
class Matrix3;
class Lattice;

class ParticlesNeighbor : public std::vector<particleIndex> {
public:

  void  buildIncluding(const particleIndex& pIndex,
                       const Lattice& lattice);

  void  buildExcluding(const particleIndex& pIndex,
                       const Lattice& lattice);

  void  interpolateVector(const ParticleVariable<Vector>& pVector,
                          Vector* data, 
                          Matrix3* gradient) const;

  void  interpolatedouble(const ParticleVariable<double>& pdouble,
                          double* data,
                          Vector* gradient) const;

private:
};

} //namespace MPM
} //namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__

// $Log$
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
