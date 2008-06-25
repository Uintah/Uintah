
#include <Packages/Uintah/Core/Grid/Variables/Iterator.h>

namespace Uintah
{
  std::ostream& operator<<(std::ostream& out, const Uintah::Iterator& b)
  {
    return b.put(out);
  }
}  
