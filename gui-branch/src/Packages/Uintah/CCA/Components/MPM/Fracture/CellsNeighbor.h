#ifndef __CELLSNEIGHBOR_H__
#define __CELLSNEIGHBOR_H__

#include <Core/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
class Lattice;
using namespace SCIRun;

class CellsNeighbor : public std::vector<IntVector> {
public:

void  buildIncluding(const IntVector& cellIndex,
                     const Lattice& lattice);

private:
};
} // End namespace Uintah


#endif //__CELLSNEIGHBOR_H__

