#include "VelocityMagnitude.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
VelocityMagnitude( const Expr::Tag& vel1tag,
           const Expr::Tag& vel2tag,
           const Expr::Tag& vel3tag )
: Expr::Expression<FieldT>(),
vel1t_( vel1tag ),
vel2t_( vel2tag ),
vel3t_( vel3tag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
~VelocityMagnitude()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( vel1t_ != Expr::Tag() )  exprDeps.requires_expression( vel1t_ );
  if( vel2t_ != Expr::Tag() )  exprDeps.requires_expression( vel2t_ );
  if( vel3t_ != Expr::Tag() )  exprDeps.requires_expression( vel3t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
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
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( vel1t_ != Expr::Tag() )  InpterpVel1T2FieldTOp_ = opDB.retrieve_operator<InpterpVel1T2FieldT>();
  if( vel2t_ != Expr::Tag() )  InpterpVel2T2FieldTOp_ = opDB.retrieve_operator<InpterpVel2T2FieldT>();
  if( vel3t_ != Expr::Tag() )  InpterpVel3T2FieldTOp_ = opDB.retrieve_operator<InpterpVel3T2FieldT>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& velMag = this->value();
  velMag=0.0;
  if( vel1t_ != Expr::Tag() ){
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( velMag );
    InpterpVel1T2FieldTOp_->apply_to_field( *vel1_, *tmp );
    velMag <<= *tmp * *tmp;
  }
  if( vel2t_ != Expr::Tag() ){
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( velMag );
    InpterpVel2T2FieldTOp_->apply_to_field( *vel2_, *tmp );
    velMag <<= velMag + *tmp * *tmp;
  }
  if( vel3t_ != Expr::Tag() ){
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( velMag );
    InpterpVel3T2FieldTOp_->apply_to_field( *vel3_, *tmp );
    velMag <<= velMag + *tmp * *tmp;
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
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
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::Builder::build() const
{
  return new VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>( v1t_, v2t_, v3t_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class VelocityMagnitude< SpatialOps::structured::SVolField,
                                  SpatialOps::structured::XVolField,
                                  SpatialOps::structured::YVolField,
                                  SpatialOps::structured::ZVolField >;
//==========================================================================
