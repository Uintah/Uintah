#include <CCA/Components/Wasatch/Expressions/VelEst.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT >
VelEst<FieldT>::VelEst( const Expr::Tag velTag,
                        const Expr::Tag convTermTag,
                        const Expr::TagList tauTags,
                        const Expr::Tag densityTag,
                        const Expr::Tag viscTag,
                        const Expr::Tag pressureTag,
                        const Expr::Tag timeStepTag )
  : Expr::Expression<FieldT>(),
    doX_   ( tauTags[0] != Expr::Tag()  ),
    doY_   ( tauTags[1] != Expr::Tag() ),
    doZ_   ( tauTags[2] != Expr::Tag() ),
    is3d_( doX_ && doY_ && doZ_ )
{
  this->set_gpu_runnable( true );
  this->template create_field_request(velTag, vel_);
  this->template create_field_request(convTermTag, convTerm_);
  this->template create_field_request(densityTag, density_);
  this->template create_field_request(pressureTag, pressure_);
  this->template create_field_request(timeStepTag, dt_);
  if (doX_ || doY_ || doZ_) this->template create_field_request(viscTag, visc_);
  if (doX_) this->template create_field_request(tauTags[0], tauxi_);
  if (doY_) this->template create_field_request(tauTags[1], tauyi_);
  if (doZ_) this->template create_field_request(tauTags[2], tauzi_);
}

//------------------------------------------------------------------

template< typename FieldT >
VelEst<FieldT>::~VelEst()
{}

//------------------------------------------------------------------

template< typename FieldT >
void VelEst<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  scalarInterpOp_ = opDB.retrieve_operator<ScalarInterpT>();
  
  if( doX_ ) {
    s2XFInterpOp_     = opDB.retrieve_operator<S2XFInterpT>();
    divXOp_ = opDB.retrieve_operator<DivXT>();
  }
  if( doY_ ) {
    s2YFInterpOp_     = opDB.retrieve_operator<S2YFInterpT>();
    divYOp_ = opDB.retrieve_operator<DivYT>();
  }
  if( doZ_ ) {
    s2ZFInterpOp_     = opDB.retrieve_operator<S2ZFInterpT>();
    divZOp_ = opDB.retrieve_operator<DivZT>();
  }

  gradPOp_ = opDB.retrieve_operator<GradPT>();

}

//------------------------------------------------------------------

template< typename FieldT >
void VelEst<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  result <<= 0.0;  // jcs without this, the variable density tests go haywire.

  const SVolField& rho = density_->field_ref();
  const SVolField& p = pressure_->field_ref();
  const FieldT& vel = vel_->field_ref();
  const FieldT& convTerm = convTerm_->field_ref();
  const TimeField& dt = dt_->field_ref();
  
  if( is3d_ ){ // optimize the 3D calculation since that is what we have most commonly:
    const SVolField& visc = visc_->field_ref();
    const XFace& tauxi = tauxi_->field_ref();
    const YFace& tauyi = tauyi_->field_ref();
    const ZFace& tauzi = tauzi_->field_ref();
    
    result <<= vel + dt * ( convTerm - ( 1 / (*scalarInterpOp_)(rho) )
        * ( (*divXOp_)( (*s2XFInterpOp_)(visc) * tauxi )
          + (*divYOp_)( (*s2YFInterpOp_)(visc) * tauyi )
          + (*divZOp_)( (*s2ZFInterpOp_)(visc) * tauzi )
          + (*gradPOp_)(p)
          )
        );
  }
  else{ // for 2D and 1D, things aren't as fast:
    SpatFldPtr<FieldT> tmp = SpatialFieldStore::get<FieldT>( result );
    if( doX_ ) *tmp <<=        (*divXOp_)( (*s2XFInterpOp_)( visc_->field_ref() ) * tauxi_->field_ref() );
    else                         *tmp <<= 0.0;
    if( doY_ ) *tmp <<= *tmp + (*divYOp_)( (*s2YFInterpOp_)( visc_->field_ref() ) * tauyi_->field_ref() );
    if( doZ_ ) *tmp <<= *tmp + (*divZOp_)( (*s2ZFInterpOp_)( visc_->field_ref() ) * tauzi_->field_ref() );
    result <<= vel + dt * ( convTerm - ( 1 / (*scalarInterpOp_)(rho) ) * ( *tmp + (*gradPOp_)(p) ) );
  }
}

//------------------------------------------------------------------

template< typename FieldT >
VelEst<FieldT>::Builder::Builder( const Expr::Tag& result,
                                  const Expr::Tag velTag,
                                  const Expr::Tag convTermTag,
                                  const Expr::TagList tauTags,
                                  const Expr::Tag densityTag,
                                  const Expr::Tag viscTag,
                                  const Expr::Tag pressureTag,
                                  const Expr::Tag timeStepTag )
: ExpressionBuilder(result),
  tauts_    ( tauTags     ),
  velt_     ( velTag      ),
  convTermt_( convTermTag ),
  densityt_ ( densityTag  ),
  visct_    ( viscTag     ),
  pt_       ( pressureTag ),
  tstpt_    ( timeStepTag )
{}

//------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VelEst<FieldT>::Builder::build() const
{
  return new VelEst<FieldT>( velt_, convTermt_, tauts_, densityt_, visct_, pt_, tstpt_ );
}
//------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class VelEst< SpatialOps::XVolField >;
template class VelEst< SpatialOps::YVolField >;
template class VelEst< SpatialOps::ZVolField >;
//==========================================================================
