#include <Packages/Uintah/Core/Grid/VarLabelMatlPatch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;

bool VarLabelMatlPatch::operator<(const VarLabelMatlPatch& other) const
{
  if (label_->equals(other.label_)) {
    if (matlIndex_ == other.matlIndex_)
      return patch_ < other.patch_;
    else
      return matlIndex_ < other.matlIndex_;
  }
  else {
    VarLabel::Compare comp;
    return comp(label_, other.label_);
  }
}

