#include <CCA/Components/Wasatch/Expressions/WeakConvectiveTerm.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT >
WeakConvectiveTerm<FieldT>::WeakConvectiveTerm( const Expr::Tag velTag,
                        const Expr::TagList velTags )
  : Expr::Expression<FieldT>(),
    velt_    ( velTag      ),
    velxt_   ( velTags[0]  ),
    velyt_   ( velTags[1]  ),
    velzt_   ( velTags[2]  ),
    is3d_( velxt_ != Expr::Tag() && velyt_ != Expr::Tag() && velzt_ != Expr::Tag() )
{
  this->set_gpu_runnable( true );
}

//------------------------------------------------------------------

template< typename FieldT >
WeakConvectiveTerm<FieldT>::~WeakConvectiveTerm()
{}

//------------------------------------------------------------------

template< typename FieldT >
void WeakConvectiveTerm<FieldT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{  
  exprDeps.requires_expression( velt_     );
  
  if( velxt_ != Expr::Tag() )  exprDeps.requires_expression( velxt_ );
  if( velyt_ != Expr::Tag() )  exprDeps.requires_expression( velyt_ );
  if( velzt_ != Expr::Tag() )  exprDeps.requires_expression( velzt_ );
}

//------------------------------------------------------------------

template< typename FieldT >
void WeakConvectiveTerm<FieldT>::bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& FM    = fml.field_manager<FieldT>();
  const typename Expr::FieldMgrSelector<XVolField>::type& xVolFM = fml.field_manager<XVolField>();
  const typename Expr::FieldMgrSelector<YVolField>::type& yVolFM = fml.field_manager<YVolField>();
  const typename Expr::FieldMgrSelector<ZVolField>::type& zVolFM = fml.field_manager<ZVolField>();
  
  vel_     = &FM.field_ref ( velt_ );    

  if( velxt_ != Expr::Tag() )  velx_ = &xVolFM.field_ref ( velxt_ ); 
  if( velyt_ != Expr::Tag() )  vely_ = &yVolFM.field_ref ( velyt_ ); 
  if( velzt_ != Expr::Tag() )  velz_ = &zVolFM.field_ref ( velzt_ );
}

//------------------------------------------------------------------

template< typename FieldT >
void WeakConvectiveTerm<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( velxt_ != Expr::Tag() ) {
    xInterpOp_     = opDB.retrieve_operator<XInterpT>();
    xFaceInterpOp_ = opDB.retrieve_operator<XFaceInterpT>();
    gradXOp_       = opDB.retrieve_operator<GradXT>();
  }
  if( velyt_ != Expr::Tag() ) {
    yInterpOp_     = opDB.retrieve_operator<YInterpT>();
    yFaceInterpOp_ = opDB.retrieve_operator<YFaceInterpT>();
    gradYOp_       = opDB.retrieve_operator<GradYT>();
  }
  if( velzt_ != Expr::Tag() ) {
    zInterpOp_     = opDB.retrieve_operator<ZInterpT>();
    zFaceInterpOp_ = opDB.retrieve_operator<ZFaceInterpT>();
    gradZOp_       = opDB.retrieve_operator<GradZT>();
  }
  
}

//------------------------------------------------------------------

template< typename FieldT >
void WeakConvectiveTerm<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  if( is3d_ ){ // inline everything for 3D:
    result <<= - (*xInterpOp_)(*velx_) * (*xFaceInterpOp_)( (*gradXOp_)(*vel_) )
               - (*yInterpOp_)(*vely_) * (*yFaceInterpOp_)( (*gradYOp_)(*vel_) )
               - (*zInterpOp_)(*velz_) * (*zFaceInterpOp_)( (*gradZOp_)(*vel_) );
  }
  else{ // not optimized in 2D and 1D:
    if (velxt_ != Expr::Tag()) result <<=        - (*xInterpOp_)(*velx_) * (*xFaceInterpOp_)( (*gradXOp_)(*vel_) );
    else                       result <<= 0.0;
    if (velyt_ != Expr::Tag()) result <<= result - (*yInterpOp_)(*vely_) * (*yFaceInterpOp_)( (*gradYOp_)(*vel_) );
    if (velzt_ != Expr::Tag()) result <<= result - (*zInterpOp_)(*velz_) * (*zFaceInterpOp_)( (*gradZOp_)(*vel_) );
  }
}

//------------------------------------------------------------------

template< typename FieldT >
WeakConvectiveTerm<FieldT>::Builder::Builder( const Expr::Tag& result,
                                  const Expr::Tag velTag,
                                  const Expr::TagList velTags )
    : ExpressionBuilder(result),
      velts_   ( velTags     ),
      velt_    ( velTag      )
{}

//------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
WeakConvectiveTerm<FieldT>::Builder::build() const
{
  return new WeakConvectiveTerm<FieldT>( velt_, velts_ );
}
//------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class WeakConvectiveTerm< SpatialOps::structured::XVolField >;
template class WeakConvectiveTerm< SpatialOps::structured::YVolField >;
template class WeakConvectiveTerm< SpatialOps::structured::ZVolField >;
//==========================================================================
