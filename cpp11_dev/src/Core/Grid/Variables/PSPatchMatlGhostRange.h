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


#ifndef CORE_GRID_VARIABLES_PSPATCHMATLGHOSTRANGE_H
#define CORE_GRID_VARIABLES_PSPATCHMATLGHOSTRANGE_H

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Patch.h>

namespace Uintah {

/**************************************

 struct
 PSPatchMatlGhostRange

 Patch, Material, Ghost, Range info


 GENERAL INFORMATION

 PSPatchMatlGhostRange.h

 Bryan Worthen
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


 KEYWORDS
 Patch, Material, Ghost, Range

 DESCRIPTION
 Has all the important information for identifying a particle subset
 patch, material, and ghost properties

 WARNING

 ****************************************/

struct  PSPatchMatlGhostRange
{
  PSPatchMatlGhostRange( const Patch     * patch
                       ,       int         matl
                       ,       IntVector   low
                       ,       IntVector   high
                       ,       int         dwid
                       ,       int         count = 1
                       )
  : m_patch(patch)
  , m_matl(matl)
  , m_low(low)
  , m_high(high)
  , m_dw_id(dwid)
  , m_count(count)
  {}

  PSPatchMatlGhostRange(const PSPatchMatlGhostRange& copy)
  : m_patch(copy.m_patch)
  , m_matl(copy.m_matl)
  , m_low(copy.m_low)
  , m_high(copy.m_high)
  , m_dw_id(copy.m_dw_id)
  , m_count(copy.m_count)
  {}
  
  bool operator<(const PSPatchMatlGhostRange& other) const;

  bool operator==(const PSPatchMatlGhostRange& other) const
  {
    return m_patch==other.m_patch && m_matl == other.m_matl && m_low == other.m_low && m_high == other.m_high && m_dw_id == other.m_dw_id;
  }

  bool operator!=(const PSPatchMatlGhostRange& other) const
  {
    return !operator==(other);
  }

  const Patch * m_patch;
  int           m_matl;
  IntVector     m_low;
  IntVector     m_high;
  int           m_dw_id;
  mutable int   m_count; // a count of how many times this has been created

};  

std::ostream& operator<<(std::ostream &out, const PSPatchMatlGhostRange &pmg);

} // end namespace Uintah

#endif // CORE_GRID_VARIABLES_PSPATCHMATLGHOSTRANGE_H
