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

#include "UpwindInterpolant.h"
#include "OperatorTypes.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/stencil/Stencil2.h>

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
UpwindInterpolant<SrcT,DestT>::
UpwindInterpolant()
{
  advectiveVelocity_ = NULL;
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
UpwindInterpolant<SrcT,DestT>::
set_advective_velocity( const DestT& theAdvectiveVelocity )
{
  // !!! NOT THREAD SAFE !!! USE LOCK
  advectiveVelocity_ = &theAdvectiveVelocity;
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
UpwindInterpolant<SrcT,DestT>::
~UpwindInterpolant()
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
UpwindInterpolant<SrcT,DestT>::
apply_to_field( const SrcT& src, DestT& dest )
{
  using namespace SpatialOps;
  using namespace SpatialOps::structured;
  typedef s2detail::ExtentsAndOffsets<SrcT,DestT> Extents;

  const MemoryWindow& ws = src.window_with_ghost();

  const MemoryWindow ws1( ws.glob_dim(),
                          ws.offset() + Extents::Src1Offset::int_vec(),
                          ws.extent() + Extents::Src1Extent::int_vec() + ws.has_bc()*Extents::Src1ExtentBC::int_vec(),
                          ws.has_bc(0), ws.has_bc(1), ws.has_bc(2) );

  const MemoryWindow ws2( ws.glob_dim(),
                          ws.offset() + Extents::Src2Offset::int_vec(),
                          ws.extent() + Extents::Src2Extent::int_vec() + ws.has_bc()*Extents::Src2ExtentBC::int_vec(),
                          ws.has_bc(0), ws.has_bc(1), ws.has_bc(2) );

  const MemoryWindow& wdest = dest.window_with_ghost();

  const MemoryWindow wd( wdest.glob_dim(),
                         wdest.offset() + Extents::DestOffset::int_vec(),
                         wdest.extent() + Extents::DestExtent::int_vec() + wdest.has_bc()*Extents::DestExtentBC::int_vec(),
                         wdest.has_bc(0), wdest.has_bc(1), wdest.has_bc(2) );

# ifndef NDEBUG
  assert( ws1.extent() == ws2.extent() && ws1.extent() == wd.extent() );
# endif

  // build fields using these newly created windows to do the stencil operation.
  // PAY ATTENTION to how we windowed on the destination field. This is likely
  // to work ONLY with SVol as source and X,Y,ZVol for destination fields.
  // Although the destination field is of a "different" type, we create a window
  // that is the "same size" as the source field to allow us to use a nebo assignment
  SrcT     d( wd,  dest.field_values(), ExternalStorage ); // NOTE here how we are crating a SrcT field from a DesT one.
  //This is a trick because we know that the fields in this case are of the same size
  SrcT  aVel( wd,  const_cast<DestT*>(advectiveVelocity_)->field_values(), ExternalStorage );
  SrcT    s1( ws1, const_cast<SrcT&>(src).field_values(), ExternalStorage );
  SrcT    s2( ws2, const_cast<SrcT&>(src).field_values(), ExternalStorage );

  d <<= cond( aVel > 0.0, s1  )
            ( aVel < 0.0, s2  )
            ( 0.5 * (s1 + s2) );

  advectiveVelocity_ = NULL;
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
namespace SS = SpatialOps::structured;

template class UpwindInterpolant< SS::SVolField, SS::SSurfXField >;
template class UpwindInterpolant< SS::SVolField, SS::SSurfYField >;
template class UpwindInterpolant< SS::SVolField, SS::SSurfZField >;

template class UpwindInterpolant< SS::XVolField, SS::XSurfXField >;
template class UpwindInterpolant< SS::XVolField, SS::XSurfYField >;
template class UpwindInterpolant< SS::XVolField, SS::XSurfZField >;

template class UpwindInterpolant< SS::YVolField, SS::YSurfXField >;
template class UpwindInterpolant< SS::YVolField, SS::YSurfYField >;
template class UpwindInterpolant< SS::YVolField, SS::YSurfZField >;

template class UpwindInterpolant< SS::ZVolField, SS::ZSurfXField >;
template class UpwindInterpolant< SS::ZVolField, SS::ZSurfYField >;
template class UpwindInterpolant< SS::ZVolField, SS::ZSurfZField >;
//==================================================================
