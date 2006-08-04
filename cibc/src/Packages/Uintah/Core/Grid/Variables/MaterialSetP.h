#ifndef UINTAH_HOMEBREW_MaterialSetP_H
#define UINTAH_HOMEBREW_MaterialSetP_H

#include <Packages/Uintah/Core/Util/Handle.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  typedef Handle<MaterialSet> MaterialSetP;
  typedef Handle<MaterialSubset> MaterialSubsetP;
}

#endif

