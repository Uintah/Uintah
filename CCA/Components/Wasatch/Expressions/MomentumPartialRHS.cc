#include "MomentumPartialRHS.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>



template< typename FieldT >
MomRHSPart<FieldT>::
MomRHSPart( const Expr::Tag& convFluxX,
            const Expr::Tag& convFluxY,
            const Expr::Tag& convFluxZ,
            const Expr::Tag& tauX,
            const Expr::Tag& tauY,
            const Expr::Tag& tauZ,
            const Expr::Tag& bodyForce )
  : Expr::Expression<FieldT>(),
    cfluxXt_( convFluxX ),
    cfluxYt_( convFluxY ),
    cfluxZt_( convFluxZ ),
    tauXt_( tauX ),
    tauYt_( tauY ),
    tauZt_( tauZ ),
    bodyForcet_( bodyForce ),
    emptyTag_( Expr::Tag() )
{}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHSPart<FieldT>::
~MomRHSPart()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( cfluxXt_ != emptyTag_ )  exprDeps.requires_expression( cfluxXt_ );
  if( cfluxYt_ != emptyTag_ )  exprDeps.requires_expression( cfluxYt_ );
  if( cfluxZt_ != emptyTag_ )  exprDeps.requires_expression( cfluxZt_ );
  if( tauXt_   != emptyTag_ )  exprDeps.requires_expression( tauXt_   );
  if( tauYt_   != emptyTag_ )  exprDeps.requires_expression( tauYt_   );
  if( tauZt_   != emptyTag_ )  exprDeps.requires_expression( tauZt_   );
  if( bodyForcet_!=emptyTag_)  exprDeps.requires_expression( bodyForcet_);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<XFluxT>& xfm = fml.template field_manager<XFluxT>();
  const Expr::FieldManager<YFluxT>& yfm = fml.template field_manager<YFluxT>();
  const Expr::FieldManager<ZFluxT>& zfm = fml.template field_manager<ZFluxT>();

  if( cfluxXt_ != emptyTag_ )  cFluxX_ = &xfm.field_ref(cfluxXt_);
  if( cfluxYt_ != emptyTag_ )  cFluxY_ = &yfm.field_ref(cfluxYt_);
  if( cfluxZt_ != emptyTag_ )  cFluxZ_ = &zfm.field_ref(cfluxZt_);

  if( tauXt_ != emptyTag_ )  tauX_ = &xfm.field_ref(tauXt_);
  if( tauYt_ != emptyTag_ )  tauY_ = &yfm.field_ref(tauYt_);
  if( tauZt_ != emptyTag_ )  tauZ_ = &zfm.field_ref(tauZt_);

  const Expr::FieldManager<FieldT>& volfm = fml.template field_manager<FieldT>();
  if( bodyForcet_ != emptyTag_ )  bodyForce_ = &volfm.field_ref( bodyForcet_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( cfluxXt_ != emptyTag_ || tauXt_ != emptyTag_ )  divXOp_ = opDB.retrieve_operator<DivX>();
  if( cfluxYt_ != emptyTag_ || tauYt_ != emptyTag_ )  divYOp_ = opDB.retrieve_operator<DivY>();
  if( cfluxZt_ != emptyTag_ || tauZt_ != emptyTag_ )  divZOp_ = opDB.retrieve_operator<DivZ>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;

  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( result );

  if( cfluxXt_ != emptyTag_ ){
    divXOp_->apply_to_field( *cFluxX_, *tmp );
    result <<= result - *tmp;
  }

  if( cfluxYt_ != emptyTag_ ){
    divYOp_->apply_to_field( *cFluxY_, *tmp );
    result <<= result - *tmp;
  }

  if( cfluxZt_ != emptyTag_ ){
    divZOp_->apply_to_field( *cFluxZ_, *tmp );
    result <<= result - *tmp;
  }

  if( tauXt_ != emptyTag_ ){
    divXOp_->apply_to_field( *tauX_, *tmp );
    result <<= result - *tmp;
  }

  if( tauYt_ != emptyTag_ ){
    divYOp_->apply_to_field( *tauY_, *tmp );
    result <<= result - *tmp;
  }

  if( tauZt_ != emptyTag_ ){
    divZOp_->apply_to_field( *tauZ_, *tmp );
    result <<= result - *tmp;
  }

  if( bodyForcet_ != emptyTag_ ){
    result <<= result + *bodyForce_;
  }

}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHSPart<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& convFluxX,
                  const Expr::Tag& convFluxY,
                  const Expr::Tag& convFluxZ,
                  const Expr::Tag& tauX,
                  const Expr::Tag& tauY,
                  const Expr::Tag& tauZ,
                  const Expr::Tag& bodyForce )
  : ExpressionBuilder(result),
    cfluxXt_   ( convFluxX ),
    cfluxYt_   ( convFluxY ),
    cfluxZt_   ( convFluxZ ),
    tauXt_     ( tauX      ),
    tauYt_     ( tauY      ),
    tauZt_     ( tauZ      ),
    bodyForcet_( bodyForce )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MomRHSPart<FieldT>::Builder::build() const
{
  return new MomRHSPart<FieldT>( cfluxXt_, cfluxYt_, cfluxZt_, tauXt_, tauYt_, tauZt_, bodyForcet_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHSPart< SpatialOps::structured::XVolField >;
template class MomRHSPart< SpatialOps::structured::YVolField >;
template class MomRHSPart< SpatialOps::structured::ZVolField >;
//==================================================================
