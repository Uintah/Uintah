#ifndef __PARTICLESNEIGHBOR_H__
#define __PARTICLESNEIGHBOR_H__

#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Geometry/Point.h>

#include <list>

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
class Matrix3;

using std::list;

class ParticlesNeighbor : public list<particleIndex> {
public:
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
// Revision 1.2  2000/06/05 22:30:02  tan
// Added interpolateVector and interpolatedouble for least-square approximation.
//
// Revision 1.1  2000/06/05 21:15:21  tan
// Added class ParticlesNeighbor to handle neighbor particles searching.
//
