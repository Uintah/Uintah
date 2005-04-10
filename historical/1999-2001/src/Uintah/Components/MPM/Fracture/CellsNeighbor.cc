#include "CellsNeighbor.h"

#include "Lattice.h"

namespace Uintah {
namespace MPM {

void  CellsNeighbor::buildIncluding(const IntVector& cellIndex,
                                    const Lattice& lattice)
{
  for(int ix=-1;ix<=1;++ix)
  for(int iy=-1;iy<=1;++iy)
  for(int iz=-1;iz<=1;++iz)
  {
    IntVector newCellIndex( cellIndex.x()+ix,
                            cellIndex.y()+iy,
                            cellIndex.z()+iz );
    if( lattice.containCell(newCellIndex) ) push_back(newCellIndex);
  }
}

} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.1  2000/06/05 23:59:55  tan
// Created class CellsNeighbor to handle cells neighbor computation.
//
