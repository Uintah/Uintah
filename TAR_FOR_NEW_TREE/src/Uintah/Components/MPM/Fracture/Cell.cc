#include "Cell.h"

namespace Uintah {
namespace MPM {

void  Cell::insert(const particleIndex& p)
{
  particles.push_back(p);
}

} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.3  2000/06/23 16:49:56  tan
// Added LeastSquare Approximation and Lattice for neighboring algorithm.
//
// Revision 1.2  2000/06/05 22:31:27  tan
// Added function to insert particle index into a cell.
//
// Revision 1.1  2000/06/05 19:47:01  tan
// Cell class will be designed to carray a link list of particle indexes
// in a cell.  This will facilitate the seaching of particles from a given
// cell.
//
