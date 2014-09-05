#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <algorithm>

namespace Uintah {
  
template<>
void ComputeSubset<const Patch*>::sort() {
  std::sort(items.begin(), items.end(), Patch::Compare());
}

template<>  
bool ComputeSubset<const Patch*>::compareElems(const Patch* e1,
					       const Patch* e2)
{ return Patch::Compare()(e1, e2); }
  
}
