#include "DiffusiveFlux.h"
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>  // jcs need to rework spatialops install structure

template< typename GradT >
DiffusiveFlux<GradT>::DiffusiveFlux( const Expr::Tag phiTag,
                                     const Expr::Tag coefTag,
                                     const Expr::ExpressionID& id,
                                     const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg ),
    isConstCoef_( false ),
    phiTag_ ( phiTag  ),
    coefTag_( coefTag ),
    coefVal_( 0.0 )
{}

//--------------------------------------------------------------------

template< typename GradT >
DiffusiveFlux<GradT>::DiffusiveFlux( const Expr::Tag phiTag,
                                     const double coef,
                                     const Expr::ExpressionID& id,
                                     const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg ),
    isConstCoef_( true ),
    phiTag_ ( phiTag ),
    coefTag_( "NULL", Expr::INVALID_CONTEXT ),
    coefVal_( coef )
{}

//--------------------------------------------------------------------

template< typename GradT >
DiffusiveFlux<GradT>::
~DiffusiveFlux()
{}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  if( !isConstCoef_ ) exprDeps.requires_expression( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FluxT  >& fluxFM   = fml.template field_manager<FluxT  >();
  const Expr::FieldManager<ScalarT>& scalarFM = fml.template field_manager<ScalarT>();

  phi_ = &scalarFM.field_ref( phiTag_ );
  if( !isConstCoef_ ) coef_ = &fluxFM.field_ref( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<GradT>();
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
evaluate()
{
  FluxT& result = this->value();

  gradOp_->apply_to_field( *phi_, result );  // J = grad(phi)
  if( isConstCoef_ ){
    result *= -coefVal_;  // J = -gamma * grad(phi)
  }
  else{
    result <<= result * -1.0 * *coef_;  // J =  - gamma * grad(phi)
  }
}


//====================================================================


template< typename GradT, typename InterpT >
DiffusiveFlux2<GradT,InterpT>::
DiffusiveFlux2( const Expr::Tag phiTag,
                const Expr::Tag coefTag,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>(id,reg),
    phiTag_ ( phiTag  ),
    coefTag_( coefTag )
{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
DiffusiveFlux2<GradT,InterpT>::
~DiffusiveFlux2()
{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  exprDeps.requires_expression( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ScalarT>& scalarFM = fml.template field_manager<ScalarT>();
  phi_  = &scalarFM.field_ref( phiTag_  );
  coef_ = &scalarFM.field_ref( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_   = opDB.retrieve_operator<GradT  >();
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
evaluate()
{
  FluxT& result = this->value();

  SpatialOps::SpatFldPtr<FluxT> fluxTmp = SpatialOps::SpatialFieldStore<FluxT>::self().get( result );

  gradOp_  ->apply_to_field( *phi_, *fluxTmp );  // J = grad(phi)
  interpOp_->apply_to_field( *coef_, result  );
  result *= *fluxTmp;                            // J =   gamma * grad(phi)
  result *= -1.0;                                // J = - gamma * grad(phi)
}

//--------------------------------------------------------------------


typedef Wasatch::OpTypes< Wasatch::SVolField >  SVOps;
typedef Wasatch::OpTypes< Wasatch::XVolField >  XVOps;
typedef Wasatch::OpTypes< Wasatch::YVolField >  YVOps;
typedef Wasatch::OpTypes< Wasatch::ZVolField >  ZVOps;

//==========================================================================
// Explicit template instantiation for supported versions of this expression
template class DiffusiveFlux< SVOps::GradX >;
template class DiffusiveFlux< SVOps::GradY >;
template class DiffusiveFlux< SVOps::GradZ >;

template class DiffusiveFlux< XVOps::GradX >;
template class DiffusiveFlux< XVOps::GradY >;
template class DiffusiveFlux< XVOps::GradZ >;

template class DiffusiveFlux< YVOps::GradX >;
template class DiffusiveFlux< YVOps::GradY >;
template class DiffusiveFlux< YVOps::GradZ >;

template class DiffusiveFlux< ZVOps::GradX >;
template class DiffusiveFlux< ZVOps::GradY >;
template class DiffusiveFlux< ZVOps::GradZ >;


template class DiffusiveFlux2< SVOps::GradX, SVOps::InterpC2FX >;
template class DiffusiveFlux2< SVOps::GradY, SVOps::InterpC2FY >;
template class DiffusiveFlux2< SVOps::GradZ, SVOps::InterpC2FZ >;

template class DiffusiveFlux2< XVOps::GradX, XVOps::InterpC2FX >;
template class DiffusiveFlux2< XVOps::GradY, XVOps::InterpC2FY >;
template class DiffusiveFlux2< XVOps::GradZ, XVOps::InterpC2FZ >;

template class DiffusiveFlux2< YVOps::GradX, YVOps::InterpC2FX >;
template class DiffusiveFlux2< YVOps::GradY, YVOps::InterpC2FY >;
template class DiffusiveFlux2< YVOps::GradZ, YVOps::InterpC2FZ >;

template class DiffusiveFlux2< ZVOps::GradX, ZVOps::InterpC2FX >;
template class DiffusiveFlux2< ZVOps::GradY, ZVOps::InterpC2FY >;
template class DiffusiveFlux2< ZVOps::GradZ, ZVOps::InterpC2FZ >;
//==========================================================================
