#include "ScalarRHS.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <Core/Exceptions/InvalidValue.h>

template< typename FieldT >
Expr::Tag ScalarRHS<FieldT>::resolve_field_tag( const typename ScalarRHS<FieldT>::FieldSelector field,
                                                const typename ScalarRHS<FieldT>::FieldTagInfo& info )
{
  Expr::Tag tag;
  const typename ScalarRHS<FieldT>::FieldTagInfo::const_iterator ifld = info.find( field );
  if( ifld != info.end() ) tag = ifld->second;
  return tag;
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::ScalarRHS( const FieldTagInfo& fieldTags,
                              const std::vector<Expr::Tag>& srcTags,
                              const Expr::Tag densityTag,
                              const Expr::Tag volFracTag,
                              const Expr::Tag xAreaFracTag,
                              const Expr::Tag yAreaFracTag,
                              const Expr::Tag zAreaFracTag,                                                                 
                              const bool isConstDensity )
  : Expr::Expression<FieldT>(),

    convTagX_( ScalarRHS<FieldT>::resolve_field_tag( CONVECTIVE_FLUX_X, fieldTags ) ),
    convTagY_( ScalarRHS<FieldT>::resolve_field_tag( CONVECTIVE_FLUX_Y, fieldTags ) ),
    convTagZ_( ScalarRHS<FieldT>::resolve_field_tag( CONVECTIVE_FLUX_Z, fieldTags ) ),

    diffTagX_( ScalarRHS<FieldT>::resolve_field_tag( DIFFUSIVE_FLUX_X, fieldTags ) ),
    diffTagY_( ScalarRHS<FieldT>::resolve_field_tag( DIFFUSIVE_FLUX_Y, fieldTags ) ),
    diffTagZ_( ScalarRHS<FieldT>::resolve_field_tag( DIFFUSIVE_FLUX_Z, fieldTags ) ),

    haveConvection_( convTagX_ != Expr::Tag() || convTagY_ != Expr::Tag() || convTagZ_ != Expr::Tag() ),
    haveDiffusion_ ( diffTagX_ != Expr::Tag() || diffTagY_ != Expr::Tag() || diffTagZ_ != Expr::Tag() ),

    doXDir_( convTagX_ != Expr::Tag() || diffTagX_ != Expr::Tag() ),
    doYDir_( convTagY_ != Expr::Tag() || diffTagY_ != Expr::Tag() ),
    doZDir_( convTagZ_ != Expr::Tag() || diffTagZ_ != Expr::Tag() ),

    volFracTag_( volFracTag ),    

    xAreaFracTag_( xAreaFracTag ),
    yAreaFracTag_( yAreaFracTag ),
    zAreaFracTag_( zAreaFracTag ),

    haveVolFrac_( volFracTag_ != Expr::Tag() ),
    haveXAreaFrac_( xAreaFracTag_ != Expr::Tag() ),
    haveYAreaFrac_( yAreaFracTag_ != Expr::Tag() ),
    haveZAreaFrac_( zAreaFracTag_ != Expr::Tag() ),

    densityTag_    ( densityTag     ),
    isConstDensity_( isConstDensity ),
    srcTags_       ( srcTags        )
{
  srcTags_.push_back( ScalarRHS<FieldT>::resolve_field_tag( SOURCE_TERM, fieldTags ) );
  nullify_fields();
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::~ScalarRHS()
{}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::nullify_fields()
{
  xConvFlux_ = NULL;  yConvFlux_ = NULL;  zConvFlux_ = NULL;
  xDiffFlux_ = NULL;  yDiffFlux_ = NULL;  zDiffFlux_ = NULL;
  divOpX_    = NULL;  divOpY_    = NULL;  divOpZ_    = NULL;
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<XFluxT>& xFluxFM  = fml.field_manager<XFluxT>();
  const Expr::FieldManager<YFluxT>& yFluxFM  = fml.field_manager<YFluxT>();
  const Expr::FieldManager<ZFluxT>& zFluxFM  = fml.field_manager<ZFluxT>();
  const Expr::FieldManager<FieldT>& scalarFM = fml.field_manager<FieldT>();
  const Expr::FieldManager<XVolField>& xVolFM  = fml.field_manager<XVolField>();
  const Expr::FieldManager<YVolField>& yVolFM  = fml.field_manager<YVolField>();
  const Expr::FieldManager<ZVolField>& zVolFM  = fml.field_manager<ZVolField>();
  

  if( haveConvection_ ){
    if( doXDir_ )  xConvFlux_ = &xFluxFM.field_ref( convTagX_ );
    if( doYDir_ )  yConvFlux_ = &yFluxFM.field_ref( convTagY_ );
    if( doZDir_ )  zConvFlux_ = &zFluxFM.field_ref( convTagZ_ );
  }

  if( haveDiffusion_ ){
    if( doXDir_ )  xDiffFlux_ = &xFluxFM.field_ref( diffTagX_ );
    if( doYDir_ )  yDiffFlux_ = &yFluxFM.field_ref( diffTagY_ );
    if( doZDir_ )  zDiffFlux_ = &zFluxFM.field_ref( diffTagZ_ );
  }
  
  if ( haveVolFrac_ ) {
    const Expr::FieldManager<SVolField>& densityFM = fml.template field_manager<SVolField>();
    volfrac_ = &densityFM.field_ref( volFracTag_ );
  }
  
  if ( haveXAreaFrac_ ) xareafrac_ = &xVolFM.field_ref( xAreaFracTag_ );
  if ( haveYAreaFrac_ ) yareafrac_ = &yVolFM.field_ref( yAreaFracTag_ );
  if ( haveZAreaFrac_ ) zareafrac_ = &zVolFM.field_ref( zAreaFracTag_ );
  

  srcTerm_.clear();
  for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
    if( isrc->context() != Expr::INVALID_CONTEXT ) {
      srcTerm_.push_back( &scalarFM.field_ref( *isrc ) );
      if (isConstDensity_){
        const Expr::FieldManager<SVolField>& densityFM = fml.template field_manager<SVolField>();
        rho_ = &densityFM.field_ref( densityTag_ );
      }
    }
  }
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( haveConvection_ ){
    if( doXDir_ )  exprDeps.requires_expression( convTagX_ );
    if( doYDir_ )  exprDeps.requires_expression( convTagY_ );
    if( doZDir_ )  exprDeps.requires_expression( convTagZ_ );
  }

  if( haveDiffusion_ ){
    if( doXDir_ )  exprDeps.requires_expression( diffTagX_ );
    if( doYDir_ )  exprDeps.requires_expression( diffTagY_ );
    if( doZDir_ )  exprDeps.requires_expression( diffTagZ_ );
  }
  
  if (haveVolFrac_) exprDeps.requires_expression( volFracTag_ );
  if ( haveXAreaFrac_ ) exprDeps.requires_expression( xAreaFracTag_ );
  if ( haveYAreaFrac_ ) exprDeps.requires_expression( yAreaFracTag_ );
  if ( haveZAreaFrac_ ) exprDeps.requires_expression( zAreaFracTag_ );  

  for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
    if( isrc->context() != Expr::INVALID_CONTEXT ){
      exprDeps.requires_expression( *isrc );
      if (isConstDensity_)
        exprDeps.requires_expression( densityTag_ );
    }
  }
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doXDir_ )  divOpX_ = opDB.retrieve_operator<DivX>();
  if( doYDir_ )  divOpY_ = opDB.retrieve_operator<DivY>();
  if( doZDir_ )  divOpZ_ = opDB.retrieve_operator<DivZ>();

  for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
    if( (isrc->context() != Expr::INVALID_CONTEXT) && isConstDensity_)
      densityInterpOp_ = opDB.retrieve_operator<DensityInterpT>();
  }
  if ( haveVolFrac_   ) volFracInterpOp_   = opDB.retrieve_operator<SVolToFieldTInterpT>();
  if ( haveXAreaFrac_ ) xAreaFracInterpOp_ = opDB.retrieve_operator<XVolToXFluxInterpT >();
  if ( haveYAreaFrac_ ) yAreaFracInterpOp_ = opDB.retrieve_operator<YVolToYFluxInterpT >();
  if ( haveZAreaFrac_ ) zAreaFracInterpOp_ = opDB.retrieve_operator<ZVolToZFluxInterpT >();
  
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::evaluate()
{
  using namespace SpatialOps;
  
  FieldT& rhs = this->value();
  rhs <<= 0.0;
  
  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( rhs );
  
  
  namespace SS = SpatialOps::structured;
  
  SpatialOps::SpatFldPtr<XFluxT> tmpx;
  SpatialOps::SpatFldPtr<XFluxT> xAreaFracInterpolated;
  
  SpatialOps::SpatFldPtr<YFluxT> tmpy;
  SpatialOps::SpatFldPtr<YFluxT> yAreaFracInterpolated;  
  
  SpatialOps::SpatFldPtr<ZFluxT> tmpz;
  SpatialOps::SpatFldPtr<ZFluxT> zAreaFracInterpolated;  
  
  // get a few memory windows
  if (doXDir_ && haveXAreaFrac_) {
    if (haveDiffusion_) {
      const SS::MemoryWindow& wx = xDiffFlux_->window_with_ghost();
      tmpx  = SpatialOps::SpatialFieldStore<XFluxT>::self().get( wx );
      xAreaFracInterpolated = SpatialOps::SpatialFieldStore<XFluxT>::self().get( wx );
    }
    else if (haveConvection_) {
      const SS::MemoryWindow& wx = xConvFlux_->window_with_ghost();
      tmpx  = SpatialOps::SpatialFieldStore<XFluxT>::self().get( wx );      
      xAreaFracInterpolated = SpatialOps::SpatialFieldStore<XFluxT>::self().get( wx );      
    }
    else {
      std::ostringstream msg;
      msg << "ERROR: xAreaFraction specified without convection or diffusion in Scalar RHS. Please revise your input file." << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );      
    }
  }  
  
  if (doYDir_ && haveYAreaFrac_) {
    if (haveDiffusion_) {
      const SS::MemoryWindow& wy = yDiffFlux_->window_with_ghost();
      tmpy  = SpatialOps::SpatialFieldStore<YFluxT>::self().get( wy );    
      yAreaFracInterpolated = SpatialOps::SpatialFieldStore<YFluxT>::self().get( wy );      
    }
    else if (haveConvection_) {
      const SS::MemoryWindow& wy = yConvFlux_->window_with_ghost();
      tmpy  = SpatialOps::SpatialFieldStore<YFluxT>::self().get( wy );      
      yAreaFracInterpolated = SpatialOps::SpatialFieldStore<YFluxT>::self().get( wy );            
    }
    else {
      std::ostringstream msg;
      msg << "ERROR: xAreaFraction specified without convection or diffusion Scalar RHS. Please revise your input file." << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );      
    }
  }
  
  if (doZDir_ && haveZAreaFrac_) {
    if (haveDiffusion_) {
      const SS::MemoryWindow& wz = zDiffFlux_->window_with_ghost();
      tmpz  = SpatialOps::SpatialFieldStore<ZFluxT>::self().get( wz );      
      zAreaFracInterpolated = SpatialOps::SpatialFieldStore<ZFluxT>::self().get( wz );            
    }
    else if (haveConvection_) {
      const SS::MemoryWindow& wz = zConvFlux_->window_with_ghost();
      tmpz  = SpatialOps::SpatialFieldStore<ZFluxT>::self().get( wz );      
      zAreaFracInterpolated = SpatialOps::SpatialFieldStore<ZFluxT>::self().get( wz );                  
    }
    else {
      std::ostringstream msg;
      msg << "ERROR: xAreaFraction specified without convection or diffusion Scalar RHS. Please revise your input file." << std::endl;
      throw Uintah::InvalidValue( msg.str(), __FILE__, __LINE__ );      
    }
  }
  
  // interpolate area fractions to XFLUXType
  if ( doXDir_ && haveXAreaFrac_ ) 
    xAreaFracInterpOp_->apply_to_field( *xareafrac_, *xAreaFracInterpolated );
  
  if ( doXDir_ && haveXAreaFrac_ ) 
    yAreaFracInterpOp_->apply_to_field( *yareafrac_, *yAreaFracInterpolated );
  
  if ( doXDir_ && haveXAreaFrac_ ) 
    zAreaFracInterpOp_->apply_to_field( *zareafrac_, *zAreaFracInterpolated );
  
  
  if( doXDir_ ){
    if( haveConvection_ ){
      if (haveXAreaFrac_) {          
        *tmpx <<= *xAreaFracInterpolated * *xConvFlux_;
        divOpX_->apply_to_field( *tmpx, *tmp );
        rhs <<= rhs - *tmp;        
      } else {
        divOpX_->apply_to_field( *xConvFlux_, *tmp );
        rhs <<= rhs - *tmp;
      }
    }
    if( haveDiffusion_ ){
      if (haveXAreaFrac_) {
        *tmpx <<= *xAreaFracInterpolated * *xDiffFlux_;
        divOpX_->apply_to_field( *tmpx, *tmp );
        rhs <<= rhs - *tmp;        
      } else {
        divOpX_->apply_to_field( *xDiffFlux_, *tmp );
        rhs <<= rhs - *tmp;
      }
    }
  }
  
  if( doYDir_ ){
    if( haveConvection_ ){
      if (haveYAreaFrac_) {
        *tmpy <<= *yAreaFracInterpolated * *yConvFlux_;
        divOpY_->apply_to_field( *tmpy, *tmp );
        rhs <<= rhs - *tmp;        
      } else {
        divOpY_->apply_to_field( *yConvFlux_, *tmp );
        rhs <<= rhs - *tmp;
      }
    }
    if( haveDiffusion_ ){
      if (haveYAreaFrac_) {
        *tmpy <<= *yAreaFracInterpolated * *yDiffFlux_;
        divOpY_->apply_to_field( *tmpy, *tmp );
        rhs <<= rhs - *tmp;        
      } else {        
        divOpY_->apply_to_field( *yDiffFlux_, *tmp );
        rhs <<= rhs - *tmp;
      }
    }
  }
  
  if( doZDir_ ){
    if( haveConvection_ ){
      if (haveZAreaFrac_) {
        *tmpz <<= *zAreaFracInterpolated * *zConvFlux_;
        divOpZ_->apply_to_field( *tmpz, *tmp );
        rhs <<= rhs - *tmp;        
      } else {        
        divOpZ_->apply_to_field( *zConvFlux_, *tmp );
        rhs <<= rhs - *tmp;
      }
    }
    if( haveDiffusion_ ){
      if (haveZAreaFrac_) {
        *tmpz <<= *zAreaFracInterpolated * *zDiffFlux_;
        divOpZ_->apply_to_field( *tmpz, *tmp );
        rhs <<= rhs - *tmp;        
      } else {                
        divOpZ_->apply_to_field( *zDiffFlux_, *tmp );
        rhs <<= rhs - *tmp;
      }
    }
  }
  
  
  if ( haveVolFrac_ ) volFracInterpOp_->apply_to_field( *volfrac_, *tmp );    
  
  typename SrcVec::const_iterator isrc;
  for( isrc=srcTerm_.begin(); isrc!=srcTerm_.end(); ++isrc ) {
    if (isConstDensity_) {
      const double densVal = (*rho_)[0];
      if (haveVolFrac_) rhs <<= rhs + *tmp*(**isrc / densVal );
      else  rhs <<= rhs + (**isrc / densVal );
    }
    else {
      if ( haveVolFrac_ )  rhs <<= rhs + **isrc * *tmp;
      else rhs <<= rhs + **isrc;
    }
  }
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::Builder::Builder( const Expr::Tag& result,
                                     const FieldTagInfo& fieldInfo,
                                     const std::vector<Expr::Tag>& sources,
                                     const Expr::Tag& densityTag,
                                     const Expr::Tag& volFracTag, 
                                     const Expr::Tag& xAreaFracTag,
                                     const Expr::Tag& yAreaFracTag,
                                     const Expr::Tag& zAreaFracTag,                                                            
                                     const bool isConstDensity )
  : ExpressionBuilder(result),
    info_          ( fieldInfo ),
    srcT_          ( sources ),
    volfracT_      ( volFracTag ),
    xareafracT_    ( xAreaFracTag ),
    yareafracT_    ( yAreaFracTag ),
    zareafracT_    ( zAreaFracTag ),
    densityT_      ( densityTag ),
    isConstDensity_( isConstDensity )
{}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::Builder::Builder( const Expr::Tag& result,
                                     const FieldTagInfo& fieldInfo,
                                     const Expr::Tag& densityTag,
                                     const Expr::Tag& volFracTag,
                                     const Expr::Tag& xAreaFracTag,
                                     const Expr::Tag& yAreaFracTag,
                                     const Expr::Tag& zAreaFracTag,                                                                                                
                                     const bool isConstDensity )
  : ExpressionBuilder(result),
    info_          ( fieldInfo ),
    volfracT_      ( volFracTag ),
    xareafracT_    ( xAreaFracTag ),
    yareafracT_    ( yAreaFracTag ),
    zareafracT_    ( zAreaFracTag ),
    densityT_      ( densityTag ),
    isConstDensity_( isConstDensity )
{}

//------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ScalarRHS<FieldT>::Builder::build() const
{
  return new ScalarRHS<FieldT>( info_, srcT_, densityT_, volfracT_, xareafracT_, yareafracT_, zareafracT_, isConstDensity_ );
}
//------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class ScalarRHS< SpatialOps::structured::SVolField >;
template class ScalarRHS< SpatialOps::structured::XVolField >;
template class ScalarRHS< SpatialOps::structured::YVolField >;
template class ScalarRHS< SpatialOps::structured::ZVolField >;
//==========================================================================
