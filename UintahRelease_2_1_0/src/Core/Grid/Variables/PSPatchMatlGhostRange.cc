/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Patch.h>

using namespace Uintah;

bool PSPatchMatlGhostRange::operator<(const PSPatchMatlGhostRange& other) const
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
  std::ostream& operator<<(std::ostream &out, const PSPatchMatlGhostRange &pmg)
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
