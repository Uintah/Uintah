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
  DestT    d( wd, &dest[0], ExternalStorage );
  DestT aVel( wd, &((*advectiveVelocity_)[0]), ExternalStorage );
  SrcT    s1( ws1, &src[0], ExternalStorage );
  SrcT    s2( ws2, &src[0], ExternalStorage );

  typename DestT::iterator      id  = d .begin();
  typename DestT::iterator      ide = d .end();
  typename DestT::iterator      iav = aVel.begin();
  typename SrcT::const_iterator is1 = s1.begin();
  typename SrcT::const_iterator is2 = s2.begin();
  for( ; id!=ide; ++id, ++iav, ++is1, ++is2 ){
    if     ( *iav > 0.0 ) *id = *is1;
    else if( *iav < 0.0 ) *id = *is2;
    else                  *id = 0.5*( *is1 + *is2 );
  }

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
