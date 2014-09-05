#include "ConvectiveFlux.h"

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


//------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT >
ConvectiveFlux<PhiInterpT, VelInterpT>::
ConvectiveFlux( const Expr::Tag phiTag,
                const Expr::Tag velTag,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<PhiFaceT>(id,reg), phiTag_( phiTag ), velTag_( velTag )
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
Expr::ExpressionBase* ConvectiveFlux<PhiInterpT, VelInterpT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new ConvectiveFlux<PhiInterpT,VelInterpT>( phiT_, velT_, id, reg );
}

//====================================================================

template< typename PhiInterpT, typename VelInterpT >
ConvectiveFluxUpwind<PhiInterpT, VelInterpT>::
ConvectiveFluxUpwind( const Expr::Tag phiTag,
                      const Expr::Tag velTag,
                      const Expr::ExpressionID& id,
                      const Expr::ExpressionRegistry& reg  )
  : ConvectiveFlux<PhiInterpT, VelInterpT>(phiTag, velTag, id, reg)
{}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
ConvectiveFluxUpwind<PhiInterpT, VelInterpT>::
~ConvectiveFluxUpwind()
{}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFluxUpwind<PhiInterpT, VelInterpT>::evaluate()
{
  PhiFaceT& result = this->value();

  // note that PhiFaceT and VelFaceT should on the same mesh location
  SpatialOps::SpatFldPtr<VelFaceT> velInterp = SpatialOps::SpatialFieldStore<VelFaceT>::self().get( result );

  // move the velocity from staggered volume to phi faces
  this->velInterpOp_->apply_to_field( *this->vel_, *velInterp );

  this->phiInterpOp_->set_advective_velocity( *velInterp );
  this->phiInterpOp_->apply_to_field( *this->phi_, result );

  result *= *velInterp;
}

//--------------------------------------------------------------------

template< typename FieldT >
struct InterpT
{
  typedef UpwindInterpolant< FieldT, typename Wasatch::FaceTypes<FieldT>::XFace >  UpwindX;
  typedef UpwindInterpolant< FieldT, typename Wasatch::FaceTypes<FieldT>::YFace >  UpwindY;
  typedef UpwindInterpolant< FieldT, typename Wasatch::FaceTypes<FieldT>::ZFace >  UpwindZ;

  typedef typename Wasatch::OpTypes<FieldT>::InterpC2FX  CentralX;
  typedef typename Wasatch::OpTypes<FieldT>::InterpC2FY  CentralY;
  typedef typename Wasatch::OpTypes<FieldT>::InterpC2FZ  CentralZ;

  typedef typename Wasatch::OperatorTypeBuilder<
    Wasatch::Interpolant,
    Wasatch::XVolField,
    typename Wasatch::FaceTypes<FieldT>::XFace>::type	VelX;

  typedef typename Wasatch::OperatorTypeBuilder<
    Wasatch::Interpolant,
    Wasatch::YVolField,
    typename Wasatch::FaceTypes<FieldT>::YFace>::type VelY;

  typedef typename Wasatch::OperatorTypeBuilder<
    Wasatch::Interpolant,
    Wasatch::ZVolField,
    typename Wasatch::FaceTypes<FieldT>::ZFace>::type VelZ;
};

typedef InterpT< Wasatch::SVolField >  SVOps;
typedef InterpT< Wasatch::XVolField >  XVOps;
typedef InterpT< Wasatch::YVolField >  YVOps;
typedef InterpT< Wasatch::ZVolField >  ZVOps;

//============================================================================
// Explicit template instantiation for supported versions of these expressions
template class ConvectiveFlux< SVOps::CentralX, SVOps::VelX >;
template class ConvectiveFlux< SVOps::CentralY, SVOps::VelY >;
template class ConvectiveFlux< SVOps::CentralZ, SVOps::VelZ >;

template class ConvectiveFlux< XVOps::CentralX, XVOps::VelX >;
template class ConvectiveFlux< XVOps::CentralY, XVOps::VelY >;
template class ConvectiveFlux< XVOps::CentralZ, XVOps::VelZ >;

template class ConvectiveFlux< YVOps::CentralX, YVOps::VelX >;
template class ConvectiveFlux< YVOps::CentralY, YVOps::VelY >;
template class ConvectiveFlux< YVOps::CentralZ, YVOps::VelZ >;

template class ConvectiveFlux< ZVOps::CentralX, ZVOps::VelX >;
template class ConvectiveFlux< ZVOps::CentralY, ZVOps::VelY >;
template class ConvectiveFlux< ZVOps::CentralZ, ZVOps::VelZ >;


template class ConvectiveFluxUpwind< SVOps::UpwindX, SVOps::VelX >;
template class ConvectiveFluxUpwind< SVOps::UpwindY, SVOps::VelY >;
template class ConvectiveFluxUpwind< SVOps::UpwindZ, SVOps::VelZ >;

template class ConvectiveFluxUpwind< XVOps::UpwindX, XVOps::VelX >;
template class ConvectiveFluxUpwind< XVOps::UpwindY, XVOps::VelY >;
template class ConvectiveFluxUpwind< XVOps::UpwindZ, XVOps::VelZ >;

template class ConvectiveFluxUpwind< YVOps::UpwindX, YVOps::VelX >;
template class ConvectiveFluxUpwind< YVOps::UpwindY, YVOps::VelY >;
template class ConvectiveFluxUpwind< YVOps::UpwindZ, YVOps::VelZ >;

template class ConvectiveFluxUpwind< ZVOps::UpwindX, ZVOps::VelX >;
template class ConvectiveFluxUpwind< ZVOps::UpwindY, ZVOps::VelY >;
template class ConvectiveFluxUpwind< ZVOps::UpwindZ, ZVOps::VelZ >;
//============================================================================
