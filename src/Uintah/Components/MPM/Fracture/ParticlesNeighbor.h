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

  void  buildIn(const IntVector& cellIndex,const Lattice& lattice);
  
  bool  visible(particleIndex idx,
                const Point& B,
		const ParticleVariable<Point>& pX,
		const ParticleVariable<int>& pIsBroken,
		const ParticleVariable<Vector>& pCrackSurfaceNormal,
		const ParticleVariable<double>& pVolume) const;

  bool visible(particleIndex idxA,
               particleIndex idxB,
	       const ParticleVariable<Point>& pX,
	       const ParticleVariable<int>& pIsBroken,
	       const ParticleVariable<Vector>& pCrackSurfaceNormal,
	       const ParticleVariable<double>& pVolume) const;

private:
};

} //namespace MPM
} //namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__

// $Log$
// Revision 1.11  2001/01/15 22:44:46  tan
// Fixed parallel version of fracture code.
//
// Revision 1.10  2000/09/22 07:18:57  tan
// MPM code works with fracture in three point bending.
//
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
