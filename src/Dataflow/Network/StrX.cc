#include <Dataflow/Network/StrX.h>

namespace SCIRun {

std::ostream& operator<<(std::ostream& target, const StrX& toDump)
{
  target << toDump.localForm();
  return target;
}

} // End namespace SCIRun


