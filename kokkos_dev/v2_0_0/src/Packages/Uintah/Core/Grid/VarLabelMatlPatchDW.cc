#include <Packages/Uintah/Core/Grid/VarLabelMatlPatchDW.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;

bool VarLabelMatlPatchDW::operator<(const VarLabelMatlPatchDW& other) const
{
  if(dw_ < other.dw_)
    return true;
  else if(dw_ > other.dw_)
    return false;

  if(matlIndex_ < other.matlIndex_)
    return true;
  else if(matlIndex_ > other.matlIndex_)
    return false;

  if(patch_ < other.patch_)
    return true;
  else if(patch_ > other.patch_)
    return false;

  VarLabel::Compare comp;
  return comp(label_, other.label_);
}

