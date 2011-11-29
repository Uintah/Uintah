#include "DiffusiveFlux.h"

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename ScalarT, typename FluxT >
DiffusiveFlux<ScalarT, FluxT>::DiffusiveFlux( const Expr::Tag rhoTag,
                                     const Expr::Tag phiTag,
                                     const Expr::Tag coefTag,
                                     const Expr::ExpressionID& id,
                                     const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg ),
    isConstCoef_( false ),
    phiTag_ ( phiTag  ),
    coefTag_( coefTag ),
    rhoTag_ ( rhoTag  ),
    coefVal_( 0.0 )
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
DiffusiveFlux<ScalarT, FluxT>::DiffusiveFlux( const Expr::Tag rhoTag,
                                     const Expr::Tag phiTag,
                                     const double coef,
                                     const Expr::ExpressionID& id,
                                     const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg ),
    isConstCoef_( true  ),
    phiTag_ ( phiTag ),
    coefTag_( "NULL", Expr::INVALID_CONTEXT ),
    rhoTag_ ( rhoTag ),
    coefVal_( coef )
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
DiffusiveFlux<ScalarT, FluxT>::
~DiffusiveFlux()
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  if( !isConstCoef_    ) exprDeps.requires_expression( coefTag_ );
  exprDeps.requires_expression( rhoTag_  );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FluxT  >& fluxFM   = fml.template field_manager<FluxT  >();
  const Expr::FieldManager<ScalarT>& scalarFM = fml.template field_manager<ScalarT>();

  phi_ = &scalarFM.field_ref( phiTag_ );
  rho_ = &fml.template field_manager<SVolField>().field_ref( rhoTag_ );
  if( !isConstCoef_ ) coef_ = &fluxFM.field_ref( coefTag_ );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_          = opDB.retrieve_operator<GradT>();
  densityInterpOp_ = opDB.retrieve_operator<DensityInterpT>();
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
evaluate()
{
  using namespace SpatialOps;
  FluxT& result = this->value();

  gradOp_->apply_to_field( *phi_, result );  // J = grad(phi)

  if( isConstCoef_ ){
    result <<= -result * coefVal_;  // J = - gamma * grad(phi)
  }
  else{
    result <<= -result * *coef_;  // J =  - gamma * grad(phi)
  }

  SpatFldPtr<FluxT> interpRho = SpatialFieldStore<FluxT>::self().get(result);
  densityInterpOp_->apply_to_field( *rho_, *interpRho );
  result <<= result * *interpRho;               // J = - rho * gamma * grad(phi)
}


//====================================================================


template< typename ScalarT, typename FluxT >
DiffusiveFlux2<ScalarT, FluxT>::
DiffusiveFlux2( const Expr::Tag rhoTag,
                const Expr::Tag phiTag,
                const Expr::Tag coefTag,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>(id,reg),
    phiTag_ ( phiTag  ),
    coefTag_( coefTag ),
    rhoTag_ ( rhoTag  )
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
DiffusiveFlux2<ScalarT, FluxT>::
~DiffusiveFlux2()
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  exprDeps.requires_expression( coefTag_ );
  exprDeps.requires_expression( rhoTag_ );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ScalarT>& scalarFM = fml.template field_manager<ScalarT>();
  phi_  = &scalarFM.field_ref( phiTag_  );
  coef_ = &scalarFM.field_ref( coefTag_ );
  rho_ = &fml.template field_manager<SVolField>().field_ref( rhoTag_ );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_   = opDB.retrieve_operator<GradT  >();
  densityInterpOp_ = opDB.retrieve_operator<DensityInterpT>();
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
evaluate()
{
  using namespace SpatialOps;
  FluxT& result = this->value();

  SpatFldPtr<FluxT> fluxTmp = SpatialFieldStore<FluxT>::self().get( result );

  gradOp_  ->apply_to_field( *phi_, *fluxTmp );  // J = grad(phi)
  interpOp_->apply_to_field( *coef_, result  );
  result <<= -result * *fluxTmp;                 // J = - gamma * grad(phi)

  densityInterpOp_->apply_to_field( *rho_, *fluxTmp );
  result <<= result * *fluxTmp;               // J = - rho * gamma * grad(phi)
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
//
#include <spatialops/structured/FVStaggered.h>

#define DECLARE_DIFF_FLUX( VOL )                                                       \
  template class DiffusiveFlux < VOL, SpatialOps::structured::FaceTypes<VOL>::XFace >; \
  template class DiffusiveFlux < VOL, SpatialOps::structured::FaceTypes<VOL>::YFace >; \
  template class DiffusiveFlux < VOL, SpatialOps::structured::FaceTypes<VOL>::ZFace >; \
  template class DiffusiveFlux2< VOL, SpatialOps::structured::FaceTypes<VOL>::XFace >; \
  template class DiffusiveFlux2< VOL, SpatialOps::structured::FaceTypes<VOL>::YFace >; \
  template class DiffusiveFlux2< VOL, SpatialOps::structured::FaceTypes<VOL>::ZFace >;

DECLARE_DIFF_FLUX( SpatialOps::structured::SVolField );
DECLARE_DIFF_FLUX( SpatialOps::structured::XVolField );
DECLARE_DIFF_FLUX( SpatialOps::structured::YVolField );
DECLARE_DIFF_FLUX( SpatialOps::structured::ZVolField );
//
//==========================================================================
