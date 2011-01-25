#include "MomentumRHS.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FVStaggered.h>


template< typename FieldT >
MomRHS<FieldT>::
MomRHS( const Expr::Tag& pressure,
        const Expr::Tag& partRHS,
        const Expr::ExpressionID& id,
        const Expr::ExpressionRegistry& reg )
  : Expr::Expression<FieldT>(id,reg),
    pressuret_( pressure ),
    rhsPartt_( partRHS )
{}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHS<FieldT>::
~MomRHS()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( pressuret_ );
  exprDeps.requires_expression( rhsPartt_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  rhsPart_ = &fm.field_ref( rhsPartt_ );

  const Expr::FieldManager<PFieldT>& pfm = fml.template field_manager<PFieldT>();
  pressure_ = &pfm.field_ref( pressuret_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<Grad>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
evaluate()
{
  FieldT& result = this->value();
  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( result );

  gradOp_->apply_to_field( *pressure_, result );
  result += *rhsPart_;
}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHS<FieldT>::
Builder::Builder( const Expr::Tag& pressure,
                  const Expr::Tag& partRHS )
  : pressuret_( pressure ),
    rhspt_( partRHS )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MomRHS<FieldT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new MomRHS<FieldT>( pressuret_, rhspt_, id, reg );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHS< SpatialOps::structured::XVolField >;
template class MomRHS< SpatialOps::structured::YVolField >;
template class MomRHS< SpatialOps::structured::ZVolField >;
//==================================================================
