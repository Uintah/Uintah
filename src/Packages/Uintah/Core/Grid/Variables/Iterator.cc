
#include <Packages/Uintah/Core/Grid/Variables/Iterator.h>

std::ostream& operator<<(std::ostream& out, const Uintah::Iterator& b)
{
  return b.put(out);
}
  
