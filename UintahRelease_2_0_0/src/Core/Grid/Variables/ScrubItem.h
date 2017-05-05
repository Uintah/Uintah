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

#ifndef CORE_GRID_VARIABLES_SCRUBITEM_H
#define CORE_GRID_VARIABLES_SCRUBITEM_H

#include <Core/Grid/Patch.h>

namespace Uintah {

class VarLabel;

struct ScrubItem {

  ScrubItem      * m_next{nullptr};
  const VarLabel * m_label;
  int              m_matl;
  const Patch    * m_patch;
  int              m_dw;
  size_t           m_hash;
  int              m_count;

  ScrubItem( const VarLabel * l
           ,       int        m
           , const Patch    * p
           ,       int        dw
           )
      : m_label(l),
        m_matl(m),
        m_patch(p),
        m_dw(dw),
        m_count(0)
  {
    size_t ptr = (size_t)l;
    m_hash = ptr ^ (m << 3) ^ (p->getID() << 4) ^ (dw << 2);
  }

  bool operator==(const ScrubItem& d)
  {
    return m_label == d.m_label && m_matl == d.m_matl && m_patch == d.m_patch && m_dw == d.m_dw;
  }

}; // struct ScrubItem

} // namespace Uintah

#endif // CORE_GRID_VARIABLES_SCRUBITEM_H
