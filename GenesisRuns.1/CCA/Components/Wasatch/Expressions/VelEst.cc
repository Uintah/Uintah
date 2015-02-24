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
    velt_     ( velTag      ),
    convTermt_( convTermTag ),
    densityt_ ( densityTag  ),
    visct_    ( viscTag     ),
    tauxit_   ( tauTags[0]  ),
    tauyit_   ( tauTags[1]  ),
    tauzit_   ( tauTags[2]  ),
    pressuret_(pressureTag ),
    tStept_   ( timeStepTag ),
    is3d_( tauxit_ != Expr::Tag() && tauyit_ != Expr::Tag() && tauzit_ != Expr::Tag() )
{
  this->set_gpu_runnable( true );
}

//------------------------------------------------------------------

template< typename FieldT >
VelEst<FieldT>::~VelEst()
{}

//------------------------------------------------------------------

template< typename FieldT >
void VelEst<FieldT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{  
  exprDeps.requires_expression( velt_      );
  exprDeps.requires_expression( convTermt_ );
  exprDeps.requires_expression( densityt_  );
  exprDeps.requires_expression( pressuret_ );
  exprDeps.requires_expression( tStept_    );  
  
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
  const typename Expr::FieldMgrSelector<FieldT>::type& fm          = fml.field_manager<FieldT>();
  const typename Expr::FieldMgrSelector<SVolField>::type& scalarFM = fml.field_manager<SVolField>();
  const typename Expr::FieldMgrSelector<XFace>::type& xFaceFM      = fml.field_manager<XFace>();
  const typename Expr::FieldMgrSelector<YFace>::type& yFaceFM      = fml.field_manager<YFace>();
  const typename Expr::FieldMgrSelector<ZFace>::type& zFaceFM      = fml.field_manager<ZFace>();
  const typename Expr::FieldMgrSelector<TimeField>::type& tFM      = fml.field_manager<TimeField>();
  
  vel_      = &fm.field_ref ( velt_ );    
  convTerm_ = &fm.field_ref ( convTermt_ );    
  density_  = &scalarFM.field_ref ( densityt_ );    
  pressure_ = &scalarFM.field_ref ( pressuret_ );    
  tStep_    = &tFM.field_ref( tStept_      );

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

  result <<= 0.0;  // jcs without this, the variable density tests go haywire.

  if( is3d_ ){ // optimize the 3D calculation since that is what we have most commonly:
    result <<= *vel_ + *tStep_ * ( *convTerm_ - ( 1 / (*scalarInterpOp_)(*density_) )
        * ( (*divXOp_)( (*s2XFInterpOp_)(*visc_) * *tauxi_ )
          + (*divYOp_)( (*s2YFInterpOp_)(*visc_) * *tauyi_ )
          + (*divZOp_)( (*s2ZFInterpOp_)(*visc_) * *tauzi_ )
          + (*gradPOp_)(*pressure_)
          )
        );
  }
  else{ // for 2D and 1D, things aren't as fast:
    SpatFldPtr<FieldT> tmp = SpatialFieldStore::get<FieldT>( result );
    if( tauxit_ != Expr::Tag() ) *tmp <<=        (*divXOp_)( (*s2XFInterpOp_)(*visc_) * *tauxi_ );
    else                         *tmp <<= 0.0;
    if( tauyit_ != Expr::Tag() ) *tmp <<= *tmp + (*divYOp_)( (*s2YFInterpOp_)(*visc_) * *tauyi_ );
    if( tauzit_ != Expr::Tag() ) *tmp <<= *tmp + (*divZOp_)( (*s2ZFInterpOp_)(*visc_) * *tauzi_ );
    result <<= *vel_ + *tStep_ * ( *convTerm_ - ( 1 / (*scalarInterpOp_)(*density_) ) * ( *tmp + (*gradPOp_)(*pressure_) ) );
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
template class VelEst< SpatialOps::structured::XVolField >;
template class VelEst< SpatialOps::structured::YVolField >;
template class VelEst< SpatialOps::structured::ZVolField >;
//==========================================================================
