#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/Array3.h>
#include <SCICore/Geometry/Point.h>

namespace Uintah {

namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

class Cell;

class Lattice : public Array3<Cell> {
public:
        Lattice(const Patch* patch,const ParticleVariable<Point>& pX);

  bool              containCell(const IntVector& cellIndex) const;
  
  const Patch*                    getPatch();
  const ParticleVariable<Point>&  getParticlesPosition();

private:
  const Patch* d_patch;
  const ParticleVariable<Point>& d_pX;
};

} //namespace MPM
} //namespace Uintah

#endif //__LATTICE_H__

// $Log$
// Revision 1.4  2000/06/05 23:57:37  tan
// Added conainCell().
//
// Revision 1.3  2000/06/05 22:32:29  tan
// Added function to find neighbor for a given particle index.
//
// Revision 1.2  2000/06/05 19:48:48  tan
// Added d_lattice which is a Array3 data of cells in a given patch.
//
// Revision 1.1  2000/06/05 17:21:40  tan
// Lattice class will be designed to make it easier to handle the grid/particle
// relationship in a given patch and a given velocity field.
//
