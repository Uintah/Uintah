#include "InterpolateExpression.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename SrcT, typename DestT >
InterpolateExpression<SrcT, DestT>::
InterpolateExpression( const Expr::Tag& srctag )
: Expr::Expression<DestT>(),
srct_( srctag )
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
InterpolateExpression<SrcT, DestT>::
~InterpolateExpression()
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( srct_ != Expr::Tag() )   exprDeps.requires_expression( srct_ );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  if( srct_ != Expr::Tag() )  src_ = &fml.template field_manager<SrcT>().field_ref( srct_ );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( srct_ != Expr::Tag() )
    InpterpSrcT2DestTOp_ = opDB.retrieve_operator<InpterpSrcT2DestT>();
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
evaluate()
{
  using SpatialOps::operator<<=;
  DestT& destResult = this->value();
  InpterpSrcT2DestTOp_->apply_to_field(*src_, destResult);
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
InterpolateExpression<SrcT, DestT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& srctag )
: ExpressionBuilder(result),
  srct_( srctag )
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
Expr::ExpressionBase*
InterpolateExpression<SrcT, DestT>::Builder::build() const
{
  return new InterpolateExpression<SrcT, DestT>( srct_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class InterpolateExpression< SpatialOps::structured::XVolField,
                                      SpatialOps::structured::SVolField >;

template class InterpolateExpression< SpatialOps::structured::YVolField,
                                      SpatialOps::structured::SVolField >;

template class InterpolateExpression< SpatialOps::structured::ZVolField,
                                      SpatialOps::structured::SVolField >;
//==========================================================================
