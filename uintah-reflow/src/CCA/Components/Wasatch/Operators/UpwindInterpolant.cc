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
#include <CCA/Components/Wasatch/FieldAdaptor.h>

#include <spatialops/NeboStencilBuilder.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

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
# ifdef ENABLE_THREADS
  /*
   * Because this operator may be accessed by multiple expressions simultaneously,
   * there is the possibility that the advective velocity could be set by multiple
   * threads simultaneously.  Therefore, we lock this until the apply_to_field()
   * is done with the advective velocity to prevent race conditions.
   */
  mutex_.lock();
#endif
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

  typedef typename OperatorTypeBuilder<Interpolant,SrcT,DestT>::type::PointCollectionType StencilPts;
  typedef typename StencilPts::Point HighStPt;
  typedef typename StencilPts::Collection::Point LowStPt;

  typedef IndexTriplet<0,0,0>           S1Offset;
  typedef LowStPt                       S1Extent;
  typedef typename LowStPt::Negate      S2Offset;
  typedef S1Extent                      S2Extent;
  typedef typename S1Extent::Negate     DOffset;
  typedef S1Extent                      DExtent;

  const MemoryWindow& ws = src.window_with_ghost();
  const MemoryWindow ws1( ws.glob_dim(),
                          ws.offset() + S1Offset::int_vec(),
                          ws.extent() + S1Extent::int_vec() );

  const MemoryWindow ws2( ws.glob_dim(),
                          ws.offset() + S2Offset::int_vec(),
                          ws.extent() + S2Extent::int_vec() );

  const MemoryWindow& wdest = dest.window_with_ghost();
  const BoundaryCellInfo& bcd = dest.boundary_info();
  const MemoryWindow wd( wdest.glob_dim(),
                         wdest.offset() + DOffset::int_vec(),
                         wdest.extent() + S1Extent::int_vec()  );

//# ifndef NDEBUG
//  assert( ws1.extent() == ws2.extent() && ws1.extent() == wd.extent() );
//# endif

  // build fields using these newly created windows to do the stencil operation.
  // PAY ATTENTION to how we windowed on the destination field. This is likely
  // to work ONLY with SVol as source and X,Y,ZVol for destination fields.
  // Although the destination field is of a "different" type, we create a window
  // that is the "same size" as the source field to allow us to use a nebo assignment
  const MemoryType dMemType = dest.memory_device_type();  // destination memory type
  const unsigned short int dDevIdx = dest.device_index(); // destination device index
  typename DestT::value_type* destVals = dest.field_values(dMemType, dDevIdx);
  SrcT  d( wd,  bcd, dest.get_ghost_data(), destVals, ExternalStorage, dMemType, dDevIdx );

  // NOTE here how we are crating a SrcT field from a DesT one.
  //This is a trick because we know that the fields in this case are of the same size
  const MemoryType advelMemType = advectiveVelocity_->memory_device_type();  // destination memory type
  const unsigned short int advelDevIdx = advectiveVelocity_->device_index(); // destination device index
  typename DestT::value_type* velVals  = const_cast<typename DestT::value_type*>(advectiveVelocity_->field_values(advelMemType, advelDevIdx));
  const SrcT  aVel( wd, bcd, advectiveVelocity_->get_ghost_data(), velVals, ExternalStorage, advelMemType, advelDevIdx );
  const SrcT    s1( ws1, src );
  const SrcT    s2( ws2, src );

  d <<= cond( aVel > 0.0, s1  )
            ( aVel < 0.0, s2  )
            ( 0.5 * (s1 + s2) );

  advectiveVelocity_ = NULL;

# ifdef ENABLE_THREADS
  mutex_.unlock();
# endif
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
namespace SS = SpatialOps::structured;

template class UpwindInterpolant< SS::SVolField, SS::SSurfXField >;
template class UpwindInterpolant< SS::SVolField, SS::SSurfYField >;
template class UpwindInterpolant< SS::SVolField, SS::SSurfZField >;

//==================================================================
