#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Geometry/Point.h>

namespace Uintah {

namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

class Lattice {
public:
         Lattice(const ParticleVariable<Point>& pX);
                
private:
};

} //namespace MPM
} //namespace Uintah

#endif //__LATTICE_H__

// $Log$
// Revision 1.1  2000/06/05 17:21:40  tan
// Lattice class will be designed to make it easier to handle the grid/particle
// relationship in a given patch and a given velocity field.
//
