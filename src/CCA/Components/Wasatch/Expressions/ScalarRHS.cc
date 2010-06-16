//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>


//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/SpatialFieldStore.h>
#include <spatialops/SpatialOperator.h>


namespace Wasatch{

  ScalarRHS::ScalarRHS( const FieldTagInfo& fieldTags,
                        const std::vector<Expr::Tag>& srcTags,
                        const Expr::ExpressionID& id,
                        const Expr::ExpressionRegistry& reg )
    : Expr::Expression<FieldT>( id, reg ),

      convTagX_( resolve_field_tag( CONVECTIVE_FLUX_X, fieldTags ) ),
      convTagY_( resolve_field_tag( CONVECTIVE_FLUX_Y, fieldTags ) ),
      convTagZ_( resolve_field_tag( CONVECTIVE_FLUX_Z, fieldTags ) ),

      diffTagX_( resolve_field_tag( DIFFUSIVE_FLUX_X, fieldTags ) ),
      diffTagY_( resolve_field_tag( DIFFUSIVE_FLUX_Y, fieldTags ) ),
      diffTagZ_( resolve_field_tag( DIFFUSIVE_FLUX_Z, fieldTags ) ),

      haveConvection_( convTagX_ != Expr::Tag() || convTagY_ != Expr::Tag() || convTagZ_ != Expr::Tag() ),
      haveDiffusion_ ( diffTagX_ != Expr::Tag() || diffTagY_ != Expr::Tag() || diffTagZ_ != Expr::Tag() ),

      doXDir_( convTagX_ != Expr::Tag() || diffTagX_ != Expr::Tag() ),
      doYDir_( convTagY_ != Expr::Tag() || diffTagY_ != Expr::Tag() ),
      doZDir_( convTagZ_ != Expr::Tag() || diffTagZ_ != Expr::Tag() ),

      srcTags_( srcTags )
  {
    srcTags_.push_back( resolve_field_tag( SOURCE_TERM, fieldTags ) );
    nullify_fields();
  }

  //------------------------------------------------------------------

  ScalarRHS::~ScalarRHS()
  {}

  //------------------------------------------------------------------

  void
  ScalarRHS::nullify_fields()
  {
    xConvFlux_ = NULL;  yConvFlux_ = NULL;  zConvFlux_ = NULL;
    xDiffFlux_ = NULL;  yDiffFlux_ = NULL;  zDiffFlux_ = NULL;

    divOpX_ = NULL;  divOpY_ = NULL;  divOpZ_ = NULL;
  }

  //------------------------------------------------------------------

  void
  ScalarRHS::bind_fields( const Expr::FieldManagerList& fml )
  {
    const Expr::FieldManager<XFluxT>& xFluxFM  = fml.field_manager<XFluxT>();
    const Expr::FieldManager<YFluxT>& yFluxFM  = fml.field_manager<YFluxT>();
    const Expr::FieldManager<ZFluxT>& zFluxFM  = fml.field_manager<ZFluxT>();
    const Expr::FieldManager<FieldT>& scalarFM = fml.field_manager<FieldT>();

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

    srcTerm_.clear();
    for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
      if( isrc->context() != Expr::INVALID_CONTEXT )
        srcTerm_.push_back( &scalarFM.field_ref( *isrc ) );
    }
  }

  //------------------------------------------------------------------

  void
  ScalarRHS::advertise_dependents( Expr::ExprDeps& exprDeps )
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

    for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
      if( isrc->context() != Expr::INVALID_CONTEXT )
        exprDeps.requires_expression( *isrc );
    }
  }

  //------------------------------------------------------------------

  void
  ScalarRHS::bind_operators( const SpatialOps::OperatorDatabase& opDB )
  {
    if( doXDir_ )  divOpX_ = opDB.retrieve_operator<DivX>();
    if( doYDir_ )  divOpY_ = opDB.retrieve_operator<DivY>();
    if( doZDir_ )  divOpZ_ = opDB.retrieve_operator<DivZ>();
  }

  //------------------------------------------------------------------

  void
  ScalarRHS::evaluate()
  {
    FieldT& rhs = this->value();
    rhs = 0.0;

    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( rhs );

    if( doXDir_ ){
      if( haveConvection_ ){
        divOpX_->apply_to_field( *xConvFlux_, *tmp );
        rhs -= *tmp;
      }
      if( haveDiffusion_ ){
        divOpX_->apply_to_field( *xDiffFlux_, *tmp );
        rhs -= *tmp;
      }
    }

    if( doYDir_ ){
      if( haveConvection_ ){
        divOpY_->apply_to_field( *yConvFlux_, *tmp );
        rhs -= *tmp;
      }
      if( haveDiffusion_ ){
        divOpY_->apply_to_field( *yDiffFlux_, *tmp );
        rhs -= *tmp;
      }
    }

    if( doZDir_ ){
      if( haveConvection_ ){
        divOpZ_->apply_to_field( *zConvFlux_, *tmp );
        rhs -= *tmp;
      }
      if( haveDiffusion_ ){
        divOpZ_->apply_to_field( *zDiffFlux_, *tmp );
        rhs -= *tmp;
      }
    }

    for( SrcVec::const_iterator isrc=srcTerm_.begin(); isrc!=srcTerm_.end(); ++isrc ){
      rhs += **isrc;
    }

  }

  //------------------------------------------------------------------

  Expr::Tag
  ScalarRHS::resolve_field_tag( const FieldSelector field,
                                const ScalarRHS::FieldTagInfo& info )
  {
    Expr::Tag tag;
    const FieldTagInfo::const_iterator ifld = info.find( field );
    if( ifld != info.end() ){
      tag = ifld->second;
    }
    return tag;
  }

  //------------------------------------------------------------------

  ScalarRHS::Builder::Builder( const FieldTagInfo& fieldInfo,
                               const std::vector<Expr::Tag>& sources )
    : info_( fieldInfo ),
      srcT_( sources )
  {}

  ScalarRHS::Builder::Builder( const FieldTagInfo& fieldInfo )
    : info_( fieldInfo )
  {}

  //------------------------------------------------------------------

  Expr::ExpressionBase*
  ScalarRHS::Builder::build( const Expr::ExpressionID& id,
                             const Expr::ExpressionRegistry& reg ) const
  {
    return new ScalarRHS( info_, srcT_, id, reg );
  }

  //------------------------------------------------------------------

} // namespace Wasatch
