#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <algorithm>

#include <iostream> // temporary

namespace Uintah {
  
template<>
void ComputeSubset<const Patch*>::sort() {
  std::sort(items.begin(), items.end(), Patch::Compare());
}

}
