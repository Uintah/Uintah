#include "Dilatation.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
Dilatation( const Expr::Tag& vel1tag,
            const Expr::Tag& vel2tag,
            const Expr::Tag& vel3tag )
  : Expr::Expression<FieldT>(),
    vel1t_( vel1tag ),
    vel2t_( vel2tag ),
    vel3t_( vel3tag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
~Dilatation()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( vel1t_ != Expr::Tag() )  exprDeps.requires_expression( vel1t_ );
  if( vel2t_ != Expr::Tag() )  exprDeps.requires_expression( vel2t_ );
  if( vel3t_ != Expr::Tag() )  exprDeps.requires_expression( vel3t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<Vel1T>& v1fm = fml.template field_manager<Vel1T>();
  const Expr::FieldManager<Vel2T>& v2fm = fml.template field_manager<Vel2T>();
  const Expr::FieldManager<Vel3T>& v3fm = fml.template field_manager<Vel3T>();

  if( vel1t_ != Expr::Tag() )  vel1_ = &v1fm.field_ref( vel1t_ );
  if( vel2t_ != Expr::Tag() )  vel2_ = &v2fm.field_ref( vel2t_ );
  if( vel3t_ != Expr::Tag() )  vel3_ = &v3fm.field_ref( vel3t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( vel1t_ != Expr::Tag() )  vel1GradOp_ = opDB.retrieve_operator<Vel1GradT>();
  if( vel2t_ != Expr::Tag() )  vel2GradOp_ = opDB.retrieve_operator<Vel2GradT>();
  if( vel3t_ != Expr::Tag() )  vel3GradOp_ = opDB.retrieve_operator<Vel3GradT>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& dil = this->value();
  dil=0.0;
  if( vel1t_ != Expr::Tag() ){
    vel1GradOp_->apply_to_field( *vel1_, dil );
  }
  if( vel2t_ != Expr::Tag() ){
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( dil );
    vel2GradOp_->apply_to_field( *vel2_, *tmp );
    dil <<= dil + *tmp;
  }
  if( vel3t_ != Expr::Tag() ){
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( dil );
    vel3GradOp_->apply_to_field( *vel3_, *tmp );
    dil <<= dil + *tmp;
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& vel1tag,
                  const Expr::Tag& vel2tag,
                  const Expr::Tag& vel3tag )
  : ExpressionBuilder(result),
    v1t_( vel1tag ), v2t_( vel2tag ), v3t_( vel3tag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Expr::ExpressionBase*
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::Builder::build() const
{
  return new Dilatation<FieldT,Vel1T,Vel2T,Vel3T>( v1t_, v2t_, v3t_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class Dilatation< SpatialOps::structured::SVolField,
                           SpatialOps::structured::XVolField,
                           SpatialOps::structured::YVolField,
                           SpatialOps::structured::ZVolField >;
//==========================================================================
