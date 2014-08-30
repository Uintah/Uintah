#ifndef heateqn_tools_h
#define heateqn_tools_h

#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/Nebo.h>


//--------------------------------------------------------------------

template< typename GradOp,
          typename InterpOp >
void
calculate_flux( const GradOp& grad,
                const InterpOp& interp,
                const typename GradOp::SrcFieldType& temperature,
                const typename GradOp::SrcFieldType& thermCond,
                typename GradOp::DestFieldType& flux )
{
  using namespace SpatialOps;

  // grab a work array from the store
  typedef typename GradOp::DestFieldType FluxT;
  SpatialOps::SpatFldPtr<FluxT> tmp = SpatialOps::SpatialFieldStore::get<FluxT>(flux);

  // interpolate the thermal conductivity to the face
  interp.apply_to_field( thermCond, *tmp );

  // apply gradient to temperature
  grad.apply_to_field( temperature, flux );

  // multiply flux by -1 and interpolated thermal conductivity
  flux <<= -flux * *tmp;
}

//--------------------------------------------------------------------

template< typename DivOp >
void
calculate_rhs( const DivOp& div,
               const typename DivOp::SrcFieldType& flux,
               const typename DivOp::DestFieldType& rhoCp,
               typename DivOp::DestFieldType& rhs )
{
  using namespace SpatialOps;

  // grab a work array from the store
  typedef typename DivOp::DestFieldType ScalarT;
  SpatialOps::SpatFldPtr<ScalarT> tmp = SpatialOps::SpatialFieldStore::get<ScalarT>(rhs);

  div.apply_to_field( flux, *tmp );
  rhs <<= rhs - *tmp / rhoCp;
}

//--------------------------------------------------------------------

#endif // heateqn_tools_h
