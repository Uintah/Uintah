
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>


namespace Uintah
{
  std::ostream& operator<<(std::ostream& out, const Uintah::NodeIterator& b)
  {
    out << "[NodeIterator at " << *b << " of " << b.end() << ']';

    return out;
  }
}
