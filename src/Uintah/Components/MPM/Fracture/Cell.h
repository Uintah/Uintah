#ifndef __CELL_H__
#define __CELL_H__

#include <SCICore/Geometry/Point.h>
#include <Uintah/Grid/ParticleSet.h>

#include <list>

namespace Uintah {

namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::list;

class Cell {
public:
  list<particleIndex> particleList;
  
  void  insert(const particleIndex& p);
private:
};

} //namespace MPM
} //namespace Uintah

#endif //__CELL_H__

// $Log$
// Revision 1.2  2000/06/05 22:31:14  tan
// Added function to insert particle index into a cell.
//
// Revision 1.1  2000/06/05 19:46:52  tan
// Cell class will be designed to carray a link list of particle indexes
// in a cell.  This will facilitate the seaching of particles from a given
// cell.
//
