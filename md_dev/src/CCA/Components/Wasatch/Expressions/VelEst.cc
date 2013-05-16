#include "VelEst.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT >
VelEst<FieldT>::VelEst( const Expr::Tag velTag,
                        const Expr::TagList velTags,
                        const Expr::TagList tauTags,
                        const Expr::Tag densityTag,
                        const Expr::Tag viscTag,
                        const Expr::Tag pressureTag,
                        const Expr::Tag timeStepTag )
  : Expr::Expression<FieldT>(),
    velt_    ( velTag      ),
    velxt_   ( velTags[0]  ),
    velyt_   ( velTags[1]  ),
    velzt_   ( velTags[2]  ),
    tauxit_  ( tauTags[0]  ),
    tauyit_  ( tauTags[1]  ),
    tauzit_  ( tauTags[2]  ),
    densityt_( densityTag  ),
    visct_   ( viscTag     ),
    pressuret_(pressureTag ),
    tStept_  ( timeStepTag )
{
}

//------------------------------------------------------------------

template< typename FieldT >
VelEst<FieldT>::~VelEst()
{}

//------------------------------------------------------------------

template< typename FieldT >
void VelEst<FieldT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{  

  exprDeps.requires_expression( velt_     );
  exprDeps.requires_expression( densityt_ );
  exprDeps.requires_expression( pressuret_ );
  exprDeps.requires_expression( tStept_   );  
  
  if( velxt_ != Expr::Tag() )  exprDeps.requires_expression( velxt_ );
  if( velyt_ != Expr::Tag() )  exprDeps.requires_expression( velyt_ );
  if( velzt_ != Expr::Tag() )  exprDeps.requires_expression( velzt_ );

  if( tauxit_ != Expr::Tag() || tauyit_ != Expr::Tag() || tauzit_ != Expr::Tag()) 
    exprDeps.requires_expression( visct_ );
    
  if( tauxit_ != Expr::Tag() )  exprDeps.requires_expression( tauxit_ );
  if( tauyit_ != Expr::Tag() )  exprDeps.requires_expression( tauyit_ );
  if( tauzit_ != Expr::Tag() )  exprDeps.requires_expression( tauzit_ );

  
}

//------------------------------------------------------------------

template< typename FieldT >
void VelEst<FieldT>::bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& FM    = fml.field_manager<FieldT>();
  const typename Expr::FieldMgrSelector<SVolField>::type& scalarFM = fml.field_manager<SVolField>();
  const typename Expr::FieldMgrSelector<XVolField>::type& xVolFM = fml.field_manager<XVolField>();
  const typename Expr::FieldMgrSelector<YVolField>::type& yVolFM = fml.field_manager<YVolField>();
  const typename Expr::FieldMgrSelector<ZVolField>::type& zVolFM = fml.field_manager<ZVolField>();
  const typename Expr::FieldMgrSelector<XFace>::type& xFaceFM = fml.field_manager<XFace>();
  const typename Expr::FieldMgrSelector<YFace>::type& yFaceFM = fml.field_manager<YFace>();
  const typename Expr::FieldMgrSelector<ZFace>::type& zFaceFM = fml.field_manager<ZFace>();
  const typename Expr::FieldMgrSelector<double>::type& tFM   = fml.field_manager<double>();
  
  vel_     = &FM.field_ref ( velt_ );    
  density_ = &scalarFM.field_ref ( densityt_ );    
  pressure_ = &scalarFM.field_ref ( pressuret_ );    
  tStep_   = &tFM.field_ref( tStept_      );

  if( velxt_ != Expr::Tag() )  velx_ = &xVolFM.field_ref ( velxt_ ); 
  if( velyt_ != Expr::Tag() )  vely_ = &yVolFM.field_ref ( velyt_ ); 
  if( velzt_ != Expr::Tag() )  velz_ = &zVolFM.field_ref ( velzt_ ); 

  if( tauxit_ != Expr::Tag() || tauyit_ != Expr::Tag() || tauzit_ != Expr::Tag()) 
    visc_ = &scalarFM.field_ref ( visct_ );    
  
  if( tauxit_ != Expr::Tag() )  tauxi_ = &xFaceFM.field_ref ( tauxit_ );
  if( tauyit_ != Expr::Tag() )  tauyi_ = &yFaceFM.field_ref ( tauyit_ );
  if( tauzit_ != Expr::Tag() )  tauzi_ = &zFaceFM.field_ref ( tauzit_ );
  
}

//------------------------------------------------------------------

template< typename FieldT >
void VelEst<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  scalarInterpOp_ = opDB.retrieve_operator<ScalarInterpT>();

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
  
  if( tauxit_ != Expr::Tag() ) {
    s2XFInterpOp_     = opDB.retrieve_operator<S2XFInterpT>();
    divXOp_ = opDB.retrieve_operator<DivXT>();
  }
  if( tauyit_ != Expr::Tag() ) {
    s2YFInterpOp_     = opDB.retrieve_operator<S2YFInterpT>();
    divYOp_ = opDB.retrieve_operator<DivYT>();
  }
  if( tauzit_ != Expr::Tag() ) {
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
  result <<= 0.0;

  SpatFldPtr<FieldT> tmp1 = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> tmp2 = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> tmp3 = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<XFace> xftmp = SpatialFieldStore::get<XFace>( result );
  SpatFldPtr<YFace> yftmp = SpatialFieldStore::get<YFace>( result );
  SpatFldPtr<ZFace> zftmp = SpatialFieldStore::get<ZFace>( result );
  SpatFldPtr<XFace> xftmp2 = SpatialFieldStore::get<XFace>( result );
  SpatFldPtr<YFace> yftmp2 = SpatialFieldStore::get<YFace>( result );
  SpatFldPtr<ZFace> zftmp2 = SpatialFieldStore::get<ZFace>( result );
  
  if (velxt_ != Expr::Tag()) {
    *tmp1  <<= 0.0;
    *tmp2  <<= 0.0;
    *xftmp <<= 0.0;
    gradXOp_->apply_to_field( *vel_, *xftmp );          // dvel/dx
    xFaceInterpOp_->apply_to_field( *xftmp, *tmp1 );
    xInterpOp_->apply_to_field( *velx_, *tmp2 );
    result <<= result - *tmp2 * *tmp1;                 // -u*dvel/dx
  }
  if (velyt_ != Expr::Tag()) {
    *tmp1  <<= 0.0;
    *tmp2  <<= 0.0;
    *yftmp <<= 0.0;
    gradYOp_->apply_to_field( *vel_, *yftmp );          // dvel/dy
    yFaceInterpOp_->apply_to_field( *yftmp, *tmp1 );
    yInterpOp_->apply_to_field( *vely_, *tmp2 );
    result <<= result - *tmp2 * *tmp1;                 // -u*dvel/dx - v*dvel/dy
  }
  if (velzt_ != Expr::Tag()) {
    *tmp1  <<= 0.0;
    *tmp2  <<= 0.0;
    *zftmp <<= 0.0;
    gradZOp_->apply_to_field( *vel_, *zftmp );          // dvel/dz
    zFaceInterpOp_->apply_to_field( *zftmp, *tmp1 );
    zInterpOp_->apply_to_field( *velz_, *tmp2 );
    result <<= result - *tmp2 * *tmp1;                 // -u*dvel/dx - v*dvel/dy -w*dvel/dz
  }

  *tmp2 <<= 0.0;
  if (tauxit_ != Expr::Tag()) {
    *tmp1 <<= 0.0;
    *xftmp <<= 0.0;
    *xftmp2 <<= 0.0;
    s2XFInterpOp_->apply_to_field( *visc_, *xftmp);
    *xftmp2 <<= *xftmp * *tauxi_;
    divXOp_->apply_to_field( *xftmp2, *tmp1 );          //+ 2*div(mu*S_xi)
    *tmp2 <<= *tmp2 + *tmp1;
  }
  if (tauyit_ != Expr::Tag()) {
    *tmp1 <<= 0.0;
    *yftmp <<= 0.0;
    *yftmp2 <<= 0.0;
    s2YFInterpOp_->apply_to_field( *visc_, *yftmp);
    *yftmp2 <<= *yftmp * *tauyi_;
    divYOp_->apply_to_field( *yftmp2, *tmp1 );          //+ 2*div(mu*S_yi)
    *tmp2 <<= *tmp2 + *tmp1;
  }
  if (tauzit_ != Expr::Tag()) {
    *tmp1 <<= 0.0;
    *zftmp <<= 0.0;
    *zftmp2 <<= 0.0;
    s2ZFInterpOp_->apply_to_field( *visc_, *zftmp);
    *zftmp2 <<= *zftmp * *tauzi_;
    divZOp_->apply_to_field( *zftmp2, *tmp1 );          //+ 2*div(mu*S_zi)
    *tmp2 <<= *tmp2 + *tmp1;
  }  
  *tmp1 <<= 0.0;
  scalarInterpOp_->apply_to_field( *density_, *tmp1 );

  gradPOp_->apply_to_field( *pressure_, *tmp3 );
  result <<= result - (1 / *tmp1) * (*tmp2 + *tmp3 );              // -u.grad(u) - (1/rho) * div(tau) - (1/rho) * grad(p^(n-1))
  
  result <<= *vel_ + *tStep_ * result;
}

//------------------------------------------------------------------

template< typename FieldT >
VelEst<FieldT>::Builder::Builder( const Expr::Tag& result,
                                  const Expr::Tag velTag,
                                  const Expr::TagList velTags,
                                  const Expr::TagList tauTags,
                                  const Expr::Tag densityTag,
                                  const Expr::Tag viscTag,
                                  const Expr::Tag pressureTag,
                                  const Expr::Tag timeStepTag )
    : ExpressionBuilder(result),
      velt_    ( velTag      ),
      velts_   ( velTags     ),
      tauts_   ( tauTags     ),
      densityt_( densityTag  ),
      visct_   ( viscTag     ),
      pt_      ( pressureTag ),
      tstpt_   ( timeStepTag )
{}

//------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VelEst<FieldT>::Builder::build() const
{
  return new VelEst<FieldT>( velt_, velts_, tauts_, densityt_, visct_, pt_, tstpt_ );
}
//------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class VelEst< SpatialOps::structured::XVolField >;
template class VelEst< SpatialOps::structured::YVolField >;
template class VelEst< SpatialOps::structured::ZVolField >;
//==========================================================================
