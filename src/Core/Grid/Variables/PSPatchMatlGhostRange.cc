/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

//_____________________________________________________________________________
//
bool PSPatchMatlGhostRange::operator<( const PSPatchMatlGhostRange & other ) const
{
  if (m_matl == other.m_matl) {
    if (m_patch->getID() == other.m_patch->getID()) {
      if (m_low == other.m_low) {
        if (m_high == other.m_high) {
          return m_dw_id < other.m_dw_id;
        }
        else {
          return m_high < other.m_high;
        }
      }
      else {
        return m_low < other.m_low;
      }
    }
    else {
      return m_patch->getID() < other.m_patch->getID();
    }
  }
  else {
    return m_matl < other.m_matl;
  }
}

//_____________________________________________________________________________
//
namespace Uintah
{
  std::ostream& operator<<( std::ostream &out, const PSPatchMatlGhostRange &pmg)
  {
    out << "Patch: " << *pmg.m_patch << " ";
    out << "Matl: "  << pmg.m_matl   << " ";
    out << "low: "   << pmg.m_low    << " ";
    out << "high: "  << pmg.m_high   << " ";
    out << "dwid: "  << pmg.m_dw_id   << " ";
    out << "count: " << pmg.m_count  << " ";
    return out;
  }
}
