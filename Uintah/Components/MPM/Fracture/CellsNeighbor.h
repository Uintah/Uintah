#ifndef __CELLSNEIGHBOR_H__
#define __CELLSNEIGHBOR_H__

#include <SCICore/Geometry/IntVector.h>

#include <list>

namespace Uintah {
namespace MPM {

class Lattice;
using SCICore::Geometry::IntVector;
using std::list;

class CellsNeighbor : public list<IntVector> {
public:

void  buildIncluding(const IntVector& cellIndex,
                     const Lattice& lattice);

private:
};

} //namespace MPM
} //namespace Uintah

#endif //__CELLSNEIGHBOR_H__

// $Log$
// Revision 1.1  2000/06/05 23:59:32  tan
// Created class CellsNeighbor to handle cells neighbor computation.
//
