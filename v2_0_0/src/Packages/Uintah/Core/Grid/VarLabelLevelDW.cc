#include <Packages/Uintah/Core/Grid/VarLabelLevelDW.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>

using namespace Uintah;

bool VarLabelLevelDW::operator<(const VarLabelLevelDW& other) const
{
  if(level_ < other.level_)
    return true;
  else if(level_ > other.level_)
    return false;

  if(dw_ < other.dw_)
    return true;
  else if(dw_ > other.dw_)
    return false;

  VarLabel::Compare comp;
  return comp(label_, other.label_);
}
