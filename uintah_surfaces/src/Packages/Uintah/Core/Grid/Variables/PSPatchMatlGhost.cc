#include <Packages/Uintah/Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;

bool PSPatchMatlGhost::operator<(const PSPatchMatlGhost& other) const
{
  if (matl_ == other.matl_)
    if (patch_ == other.patch_)
      if (gt_ == other.gt_)
        if (numGhostCells_ == other.numGhostCells_)
          return dwid_ < other.dwid_;
        else
          return numGhostCells_ < other.numGhostCells_;
      else
        return gt_ < other.gt_;
    else
      return patch_ < other.patch_;
  else
    return matl_ < other.matl_;
}

