#include "Cell.h"

namespace Uintah {
void  Cell::insert(const particleIndex& p)
{
  particles.push_back(p);
}
} // End namespace Uintah


