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

class Lattice {
public:
         Lattice(const Patch* patch,const ParticleVariable<Point>& pX);
                
private:
  Array3<Cell>  d_lattice;
};

} //namespace MPM
} //namespace Uintah

#endif //__LATTICE_H__

// $Log$
// Revision 1.2  2000/06/05 19:48:48  tan
// Added d_lattice which is a Array3 data of cells in a given patch.
//
// Revision 1.1  2000/06/05 17:21:40  tan
// Lattice class will be designed to make it easier to handle the grid/particle
// relationship in a given patch and a given velocity field.
//
