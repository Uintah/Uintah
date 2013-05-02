/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#include "Extrapolant.h"
#include "OperatorTypes.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/stencil/Stencil2.h>

//--------------------------------------------------------------------

template< typename FieldT >
Extrapolant<FieldT>::
Extrapolant()
{}

//--------------------------------------------------------------------

template< typename FieldT >
Extrapolant<FieldT>::
~Extrapolant()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Extrapolant<FieldT>::
apply_to_field(FieldT& src )
{
  // extrapolate from interior cells:
  using namespace SpatialOps;
  using namespace SpatialOps::structured;
  
  const MemoryWindow& ws = src.window_with_ghost();
  
  std::vector<IntVec> ijkShift;
  ijkShift.push_back(IntVec(1,0,0));
  ijkShift.push_back(IntVec(0,1,0));
  ijkShift.push_back(IntVec(0,0,1));
  
  int pm[2]={1,-1}; // plus or minus face
  int zo[2]={0,1};  // zero and one
  
  for (int face=0; face<3; face++) {
    for (int direc=0; direc<2; direc++) {
      IntVec extent = ws.extent() - ijkShift[face]*ws.glob_dim() + ijkShift[face];
      IntVec baseOffset = ws.offset() + (ijkShift[face]*ws.glob_dim() - ijkShift[face] )* zo[direc];
      
      const MemoryWindow wd( ws.glob_dim(),
                            baseOffset,
                            extent,
                            ws.has_bc(0), ws.has_bc(1), ws.has_bc(2) );
      
      const MemoryWindow ws1( ws.glob_dim(),
                             baseOffset + ijkShift[face] * pm[direc],
                             extent,
                             ws.has_bc(0), ws.has_bc(1), ws.has_bc(2) );
      
      const MemoryWindow ws2( ws.glob_dim(),
                             baseOffset  + ijkShift[face] * pm[direc] * 2,
                             extent,
                             ws.has_bc(0), ws.has_bc(1), ws.has_bc(2) );
      
      FieldT d  ( wd,  &src[0], ExternalStorage);
      FieldT s1 ( ws1, &src[0], ExternalStorage);
      FieldT s2 ( ws2, &src[0], ExternalStorage);
      d <<= 2.0 * s1 - s2;
    }
  }
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
namespace SS = SpatialOps::structured;
template class Extrapolant<SS::SVolField>;
template class Extrapolant<SS::XVolField>;
template class Extrapolant<SS::YVolField>;
template class Extrapolant<SS::ZVolField>;
//==================================================================
