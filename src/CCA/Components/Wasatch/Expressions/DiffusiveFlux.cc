#include "DiffusiveFlux.h"

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
  using namespace SpatialOps;
  FluxT& result = this->value();

  gradOp_->apply_to_field( *phi_, result );  // J = grad(phi)
  if( isConstCoef_ ){
    result *= -coefVal_;  // J = -gamma * grad(phi)
  }
  else{
    result <<= -result * *coef_;  // J =  - gamma * grad(phi)
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
  using namespace SpatialOps;
  FluxT& result = this->value();

  SpatFldPtr<FluxT> fluxTmp = SpatialFieldStore<FluxT>::self().get( result );

  gradOp_  ->apply_to_field( *phi_, *fluxTmp );  // J = grad(phi)
  interpOp_->apply_to_field( *coef_, result  );
  result <<= -result * *fluxTmp;                 // J = - gamma * grad(phi)
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
//
#include <spatialops/structured/FVStaggered.h>

#define DECLARE_DIFF_FLUX( VOL )								\
  template class DiffusiveFlux< SpatialOps::structured::BasicOpTypes<VOL>::GradX >; 		\
  template class DiffusiveFlux< SpatialOps::structured::BasicOpTypes<VOL>::GradY >;		\
  template class DiffusiveFlux< SpatialOps::structured::BasicOpTypes<VOL>::GradZ >;		\
  template class DiffusiveFlux2< SpatialOps::structured::BasicOpTypes<VOL>::GradX,		\
                                 SpatialOps::structured::BasicOpTypes<VOL>::InterpC2FX >;       \
  template class DiffusiveFlux2< SpatialOps::structured::BasicOpTypes<VOL>::GradY,		\
                                 SpatialOps::structured::BasicOpTypes<VOL>::InterpC2FY >;	\
  template class DiffusiveFlux2< SpatialOps::structured::BasicOpTypes<VOL>::GradZ,		\
                                 SpatialOps::structured::BasicOpTypes<VOL>::InterpC2FZ >;

DECLARE_DIFF_FLUX( SpatialOps::structured::SVolField );
DECLARE_DIFF_FLUX( SpatialOps::structured::XVolField );
DECLARE_DIFF_FLUX( SpatialOps::structured::YVolField );
DECLARE_DIFF_FLUX( SpatialOps::structured::ZVolField );
//
//==========================================================================
