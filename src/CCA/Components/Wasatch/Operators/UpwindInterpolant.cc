/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
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
  advectiveVelocity_ = nullptr;
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
# endif
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

  typedef typename OperatorTypeBuilder<Interpolant,SrcT,DestT>::type::PointCollectionType StencilPts;
  typedef typename StencilPts::Collection::Point LowStPt;
  typedef typename StencilPts::Collection::Last  HiStPt;  // assumes a 2-point stencil

  typedef LowStPt  S1Shift;  // relative to dest
  typedef HiStPt   S2Shift;  // relative to dest

  typedef IndexTriplet<0,0,0>                         S1Offset;
  typedef typename Add     <S1Offset,S1Shift>::result S1Extent;
  typedef typename Subtract<S1Offset,S1Shift>::result DOffset ;
  typedef typename Add     <S1Offset,S1Shift>::result DExtent ;
  typedef typename Subtract<S1Offset,S2Shift>::result S2Offset;
  typedef S1Extent                                    S2Extent;

  const MemoryWindow& ws = src.window_with_ghost();
  const MemoryWindow ws1( ws.glob_dim(),
                          ws.offset() + S1Offset::int_vec(),
                          ws.extent() + S1Extent::int_vec() );

  const MemoryWindow ws2( ws.glob_dim(),
                          ws.offset() + S2Offset::int_vec(),
                          ws.extent() + S2Extent::int_vec() );

  const MemoryWindow& wdest = dest.window_with_ghost();

  const MemoryWindow wd( wdest.glob_dim(),
                         wdest.offset() + DOffset::int_vec(),
                         ws.extent() + DExtent::int_vec() );  // yes, ws.

  const BoundaryCellInfo& bcs =  src.boundary_info();

# ifndef NDEBUG
  assert( ws1.extent() == ws2.extent() );
  assert( ws1.extent() ==  wd.extent() );
  assert( ws2.extent() ==  wd.extent() );
# endif

  // build fields using these newly created windows to do the stencil operation.
  // PAY ATTENTION to how we windowed on the destination field. This is likely
  // to work ONLY with SVol as source and X,Y,ZVol for destination fields.
  // Although the destination field is of a "different" type, we create a window
  // that is the "same size" as the source field to allow us to use a nebo assignment
  const short int dDevIdx = dest.active_device_index(); // destination device index
  typename DestT::value_type* destVals = dest.field_values(dDevIdx);
  SrcT d( wd, bcs, dest.get_ghost_data(), destVals, ExternalStorage, dDevIdx );

  // NOTE here how we are crating a SrcT field from a DesT one.
  //This is a trick because we know that the fields in this case are of the same size
  const short int advelDevIdx = advectiveVelocity_->active_device_index(); // destination device index
  typename DestT::value_type* velVals = const_cast<typename DestT::value_type*>(advectiveVelocity_->field_values(advelDevIdx));
  const SrcT  aVel( wd, bcs, advectiveVelocity_->get_ghost_data(), velVals, ExternalStorage, advelDevIdx );
  const SrcT    s1( ws1, src );
  const SrcT    s2( ws2, src );

  d <<= cond( aVel > 0.0, s1  )
            ( aVel < 0.0, s2  )
            ( 0.5 * (s1 + s2) );

  advectiveVelocity_ = nullptr;

# ifdef ENABLE_THREADS
  mutex_.unlock();
# endif
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class UpwindInterpolant< SpatialOps::SVolField, SpatialOps::SSurfXField >;
template class UpwindInterpolant< SpatialOps::SVolField, SpatialOps::SSurfYField >;
template class UpwindInterpolant< SpatialOps::SVolField, SpatialOps::SSurfZField >;

template class UpwindInterpolant< SpatialOps::XVolField, SpatialOps::XSurfXField >;
template class UpwindInterpolant< SpatialOps::XVolField, SpatialOps::XSurfYField >;
template class UpwindInterpolant< SpatialOps::XVolField, SpatialOps::XSurfZField >;

template class UpwindInterpolant< SpatialOps::YVolField, SpatialOps::YSurfXField >;
template class UpwindInterpolant< SpatialOps::YVolField, SpatialOps::YSurfYField >;
template class UpwindInterpolant< SpatialOps::YVolField, SpatialOps::YSurfZField >;

template class UpwindInterpolant< SpatialOps::ZVolField, SpatialOps::ZSurfXField >;
template class UpwindInterpolant< SpatialOps::ZVolField, SpatialOps::ZSurfYField >;
template class UpwindInterpolant< SpatialOps::ZVolField, SpatialOps::ZSurfZField >;

//==================================================================
