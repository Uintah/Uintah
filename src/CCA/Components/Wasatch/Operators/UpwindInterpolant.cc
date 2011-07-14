#include "UpwindInterpolant.h"
#include "OperatorTypes.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/stencil/Stencil2.h>

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
UpwindInterpolant<PhiVolT,PhiFaceT>::
UpwindInterpolant()
{
  advectiveVelocity_ = NULL;
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
UpwindInterpolant<PhiVolT,PhiFaceT>::
set_advective_velocity( const PhiFaceT& theAdvectiveVelocity )
{
  // !!! NOT THREAD SAFE !!! USE LOCK
  advectiveVelocity_ = &theAdvectiveVelocity;
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
UpwindInterpolant<PhiVolT,PhiFaceT>::
~UpwindInterpolant()
{}  

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void 
UpwindInterpolant<PhiVolT,PhiFaceT>::
apply_to_field( const PhiVolT& src, PhiFaceT& dest )
{
  using SpatialOps::structured::IntVec;

  if( advectiveVelocity_ == NULL ){
    std::ostringstream msg;
    msg << "ERROR: advecting velocity has not been set in UpwindInterpolant." << std::endl
        << "       be sure to call set_advective_velocity() prior to apply_to_field()" << std::endl;
    throw std::runtime_error( msg.str() );
  }

  const SpatialOps::structured::Stencil2Helper<PhiVolT,PhiFaceT>
    helper( src.window_with_ghost(), dest.window_with_ghost() );

  const IntVec sinc = helper.src_increment ();
  const IntVec dinc = helper.dest_increment();

  typename PhiFaceT::iterator       idest  = dest.begin() + helper.dest_offset();
  typename PhiFaceT::const_iterator advVel = advectiveVelocity_->begin() + helper.dest_offset();
  typename PhiVolT ::const_iterator isrcm  = src.begin() + helper.src_offset_lo();
  typename PhiVolT ::const_iterator isrcp  = src.begin() + helper.src_offset_hi();

  const IntVec lo = helper.low ();
  const IntVec hi = helper.high();

  for( int k=lo[2]; k<hi[2]; ++k ){
    for( int j=lo[1]; j<hi[1]; ++j ){
      for( int i=lo[0]; i<hi[0]; ++i ){
        
        if( *advVel > 0.0 ){
          *idest = *isrcm;
        }
        else if( *advVel < 0.0 ){
          *idest = *isrcp;
        }
        else{
          *idest = 0.5 * ( *isrcp + *isrcm );
        }

        advVel += dinc[0];
        idest  += dinc[0];
        isrcm  += sinc[0];
        isrcp  += sinc[0];
      }
      advVel += dinc[1];
      idest  += dinc[1];
      isrcm  += sinc[1];
      isrcp  += sinc[1];
    }
    advVel += dinc[2];
    idest  += dinc[2];
    isrcm  += sinc[2];
    isrcp  += sinc[2];
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
