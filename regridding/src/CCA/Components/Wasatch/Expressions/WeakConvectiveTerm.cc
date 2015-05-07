#include <CCA/Components/Wasatch/Expressions/WeakConvectiveTerm.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT >
WeakConvectiveTerm<FieldT>::WeakConvectiveTerm( const Expr::Tag velTag,
                        const Expr::TagList velTags )
  : Expr::Expression<FieldT>(),
    doX_   ( velTags[0] != Expr::Tag()  ),
    doY_   ( velTags[1] != Expr::Tag()  ),
    doZ_   ( velTags[2] != Expr::Tag() ),
    is3d_( doX_ && doY_ && doZ_ )
{
  this->set_gpu_runnable( true );
   vel_ = this->template create_field_request<FieldT>(velTag);
  if (doX_)  u_ = this->template create_field_request<XVolField>(velTags[0]);
  if (doY_)  v_ = this->template create_field_request<YVolField>(velTags[1]);
  if (doZ_)  w_ = this->template create_field_request<ZVolField>(velTags[2]);
}

//------------------------------------------------------------------

template< typename FieldT >
WeakConvectiveTerm<FieldT>::~WeakConvectiveTerm()
{}

//------------------------------------------------------------------

template< typename FieldT >
void WeakConvectiveTerm<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doX_ ) {
    xInterpOp_     = opDB.retrieve_operator<XInterpT>();
    xFaceInterpOp_ = opDB.retrieve_operator<XFaceInterpT>();
    gradXOp_       = opDB.retrieve_operator<GradXT>();
  }
  if( doY_ ) {
    yInterpOp_     = opDB.retrieve_operator<YInterpT>();
    yFaceInterpOp_ = opDB.retrieve_operator<YFaceInterpT>();
    gradYOp_       = opDB.retrieve_operator<GradYT>();
  }
  if( doZ_ ) {
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
  const FieldT& vel = vel_->field_ref();
  
  if( is3d_ ){ // inline everything for 3D:
    const XVolField& u = u_->field_ref();
    const YVolField& v = v_->field_ref();
    const ZVolField& w = w_->field_ref();
    result <<= - (*xInterpOp_)(u) * (*xFaceInterpOp_)( (*gradXOp_)(vel) )
               - (*yInterpOp_)(v) * (*yFaceInterpOp_)( (*gradYOp_)(vel) )
               - (*zInterpOp_)(w) * (*zFaceInterpOp_)( (*gradZOp_)(vel) );
  }
  else{ // not optimized in 2D and 1D:
    if (doX_) result <<=        - (*xInterpOp_)( u_->field_ref() ) * (*xFaceInterpOp_)( (*gradXOp_)(vel) );
    else                       result <<= 0.0;
    if (doY_) result <<= result - (*yInterpOp_)( v_->field_ref() ) * (*yFaceInterpOp_)( (*gradYOp_)(vel) );
    if (doZ_) result <<= result - (*zInterpOp_)( w_->field_ref() ) * (*zFaceInterpOp_)( (*gradZOp_)(vel) );
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
template class WeakConvectiveTerm< SpatialOps::XVolField >;
template class WeakConvectiveTerm< SpatialOps::YVolField >;
template class WeakConvectiveTerm< SpatialOps::ZVolField >;
//==========================================================================
