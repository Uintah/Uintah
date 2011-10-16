#include "ConvectiveFlux.h"

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/FieldExpressions.h>

//------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT >
ConvectiveFlux<PhiInterpT, VelInterpT>::
ConvectiveFlux( const Expr::Tag phiTag,
                const Expr::Tag velTag,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<PhiFaceT>(id,reg),
    phiTag_( phiTag ),
    velTag_( velTag )
{}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
ConvectiveFlux<PhiInterpT, VelInterpT>::
~ConvectiveFlux()
{}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression(phiTag_);
  exprDeps.requires_expression(velTag_);
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<PhiVolT>& phiVolFM = fml.template field_manager<PhiVolT>();
  phi_ = &phiVolFM.field_ref( phiTag_ );
  
  const Expr::FieldManager<VelVolT>& velVolFM = fml.template field_manager<VelVolT>();
  vel_ = &velVolFM.field_ref( velTag_ );
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  velInterpOp_ = opDB.retrieve_operator<VelInterpT>();
  phiInterpOp_ = opDB.retrieve_operator<PhiInterpT>();
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::evaluate()
{
  PhiFaceT& result = this->value();

  // note that PhiFaceT and VelFaceT should on the same mesh location
  SpatialOps::SpatFldPtr<VelFaceT> velInterp = SpatialOps::SpatialFieldStore<VelFaceT>::self().get( result );

  // move the velocity from staggered volume to phi faces
  velInterpOp_->apply_to_field( *vel_, *velInterp );

  // intepolate phi to the control volume faces
  phiInterpOp_->apply_to_field( *phi_, result );

  result *= *velInterp;
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
Expr::ExpressionBase*
ConvectiveFlux<PhiInterpT, VelInterpT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new ConvectiveFlux<PhiInterpT,VelInterpT>( phiT_, velT_, id, reg );
}

//====================================================================

template< typename PhiInterpT, typename VelInterpT >
ConvectiveFluxLimiter<PhiInterpT, VelInterpT>::
ConvectiveFluxLimiter( const Expr::Tag phiTag,
                       const Expr::Tag velTag,
                       const Wasatch::ConvInterpMethods limiterType,
                       const Expr::ExpressionID& id,
                       const Expr::ExpressionRegistry& reg )
  : Expr::Expression<PhiFaceT>(id,reg),
    phiTag_( phiTag ),
    velTag_( velTag ),
    limiterType_( limiterType )
{}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
ConvectiveFluxLimiter<PhiInterpT, VelInterpT>::
~ConvectiveFluxLimiter()
{}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void
ConvectiveFluxLimiter<PhiInterpT, VelInterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression(phiTag_);
  exprDeps.requires_expression(velTag_);
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void
ConvectiveFluxLimiter<PhiInterpT, VelInterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<PhiVolT>& phiVolFM = fml.template field_manager<PhiVolT>();
  phi_ = &phiVolFM.field_ref( phiTag_ );
  
  const Expr::FieldManager<VelVolT>& velVolFM = fml.template field_manager<VelVolT>();
  vel_ = &velVolFM.field_ref( velTag_ );
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void
ConvectiveFluxLimiter<PhiInterpT, VelInterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  velInterpOp_ = opDB.retrieve_operator<VelInterpT>();
  phiInterpOp_ = opDB.retrieve_operator<PhiInterpT>();
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void
ConvectiveFluxLimiter<PhiInterpT, VelInterpT>::evaluate()
{
  using namespace SpatialOps;
  PhiFaceT& result = this->value();
  
  // note that PhiFaceT and VelFaceT should on the same mesh location
  SpatialOps::SpatFldPtr<VelFaceT> velInterp = SpatialOps::SpatialFieldStore<VelFaceT>::self().get( result );
  
  // move the velocity from staggered volume to phi faces
  velInterpOp_->apply_to_field( *vel_, *velInterp );

  phiInterpOp_->set_advective_velocity( *velInterp );
  phiInterpOp_->set_flux_limiter_type( limiterType_ );
  phiInterpOp_->apply_to_field( *phi_, result );
  
  result <<= result * *velInterp;
}

//--------------------------------------------------------------------


//============================================================================
// Explicit template instantiation for supported versions of these expressions
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>

using Wasatch::OpTypes;

#define CONV_FLUX_DECLARE( VOL )                                        \
  template class ConvectiveFlux< OpTypes<VOL>::InterpC2FX, OperatorTypeBuilder<Interpolant,XVolField,FaceTypes<VOL>::XFace>::type >; \
  template class ConvectiveFlux< OpTypes<VOL>::InterpC2FY, OperatorTypeBuilder<Interpolant,YVolField,FaceTypes<VOL>::YFace>::type >; \
  template class ConvectiveFlux< OpTypes<VOL>::InterpC2FZ, OperatorTypeBuilder<Interpolant,ZVolField,FaceTypes<VOL>::ZFace>::type >;

CONV_FLUX_DECLARE( SVolField );
CONV_FLUX_DECLARE( XVolField );
CONV_FLUX_DECLARE( YVolField );
CONV_FLUX_DECLARE( ZVolField );

#define CONV_FLUX_DECLARE_UPW( VOL )                                    \
  template class ConvectiveFluxLimiter< OpTypes<VOL>::InterpC2FXUpwind, OperatorTypeBuilder<Interpolant,XVolField,FaceTypes<VOL>::XFace>::type >; \
  template class ConvectiveFluxLimiter< OpTypes<VOL>::InterpC2FYUpwind, OperatorTypeBuilder<Interpolant,YVolField,FaceTypes<VOL>::YFace>::type >; \
  template class ConvectiveFluxLimiter< OpTypes<VOL>::InterpC2FZUpwind, OperatorTypeBuilder<Interpolant,ZVolField,FaceTypes<VOL>::ZFace>::type >;

CONV_FLUX_DECLARE_UPW( SVolField );
CONV_FLUX_DECLARE_UPW( XVolField );
CONV_FLUX_DECLARE_UPW( YVolField );
CONV_FLUX_DECLARE_UPW( ZVolField );

#define CONV_FLUX_LIMITER_DECLARE_LIMITER( VOL )                        \
  template class ConvectiveFluxLimiter< OpTypes<VOL>::InterpC2FXLimiter, OperatorTypeBuilder<Interpolant,XVolField,FaceTypes<VOL>::XFace>::type >; \
  template class ConvectiveFluxLimiter< OpTypes<VOL>::InterpC2FYLimiter, OperatorTypeBuilder<Interpolant,YVolField,FaceTypes<VOL>::YFace>::type >; \
  template class ConvectiveFluxLimiter< OpTypes<VOL>::InterpC2FZLimiter, OperatorTypeBuilder<Interpolant,ZVolField,FaceTypes<VOL>::ZFace>::type >;

CONV_FLUX_LIMITER_DECLARE_LIMITER( SVolField );
CONV_FLUX_LIMITER_DECLARE_LIMITER( XVolField );
CONV_FLUX_LIMITER_DECLARE_LIMITER( YVolField );
CONV_FLUX_LIMITER_DECLARE_LIMITER( ZVolField );


//============================================================================
