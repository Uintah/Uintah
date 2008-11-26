#include <Packages/Uintah/Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;

bool PSPatchMatlGhost::operator<(const PSPatchMatlGhost& other) const
{
  if (matl_ == other.matl_)
    if (patch_->getID() == other.patch_->getID())
      if (low_ == other.low_)
        if (high_ == other.high_)
          return dwid_ < other.dwid_;
        else
          return high_ < other.high_;
      else
        return low_ < other.low_;
    else
      return patch_->getID() < other.patch_->getID();
  else
    return matl_ < other.matl_;
}
namespace Uintah
{
  ostream& operator<<(ostream &out, const PSPatchMatlGhost &pmg)
  {
    out << "Patch: " << *pmg.patch_ << " ";
    out << "Matl: " << pmg.matl_ << " ";
    out << "low: " << pmg.low_ << " ";
    out << "high: " << pmg.high_ << " ";
    out << "dwid: " << pmg.dwid_ << " ";
    out << "count: " << pmg.count_ << " ";
    return out;
  }
}
