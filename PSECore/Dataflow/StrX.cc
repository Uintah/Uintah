#include <PSECore/Dataflow/StrX.h>

namespace PSECore {
namespace Dataflow {

std::ostream& operator<<(std::ostream& target, const StrX& toDump)
{
  target << toDump.localForm();
  return target;
}

} // Dataflow
} // PSECore


