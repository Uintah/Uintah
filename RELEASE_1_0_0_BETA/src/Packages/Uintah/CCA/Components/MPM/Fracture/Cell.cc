#include "Cell.h"

namespace Uintah {

void  Cell::insert(const particleIndex& p)
{
  particles.push_back(p);
}

void  Cell::insert(const CrackFace& crackFace)
{
  crackFaces.push_back(crackFace);
}

} // End namespace Uintah


