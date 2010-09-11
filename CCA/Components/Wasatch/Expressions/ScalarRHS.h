#ifndef ScalarRHS_h
#define ScalarRHS_h

#include <map>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/SpatialFieldStore.h>


namespace Wasatch{

  /**
   *  \ingroup WasatchExpressions
   *  \class ScalarRHS
   *  \author James C. Sutherland
   *
   *  \brief Support for a basic scalar transport equation involving
   *         any/all of advection, diffusion and reaction.
   *
   *  The ScalarRHS Expression defines a template class for basic
   *  transport equations.  Each equation is templated on an interpolant
   *  and divergence operator, from which the field types are deduced.
   *
   *  The user provides expressions to calculate the advecting velocity,
   *  diffusive fluxes and/or source terms.  This will then calculate
   *  the full RHS for use with the time integrator.
   */
  template< typename FieldT >
  class ScalarRHS : public Expr::Expression<FieldT>
  {
  protected:

    typedef typename Wasatch::FaceTypes<FieldT>::XFace XFluxT; ///< The type of field for the x-face variables.
    typedef typename Wasatch::FaceTypes<FieldT>::YFace YFluxT; ///< The type of field for the y-face variables.
    typedef typename Wasatch::FaceTypes<FieldT>::ZFace ZFluxT; ///< The type of field for the z-face variables.

    typedef typename Wasatch::OpTypes<FieldT>::DivX   DivX; ///< Divergence operator (surface integral) in the x-direction
    typedef typename Wasatch::OpTypes<FieldT>::DivY   DivY; ///< Divergence operator (surface integral) in the y-direction
    typedef typename Wasatch::OpTypes<FieldT>::DivZ   DivZ; ///< Divergence operator (surface integral) in the z-direction

  public:

    /**
     *  \enum FieldSelector
     *  \brief Use this enum to populate information in the FieldTagInfo type.
     */
    enum FieldSelector{
      CONVECTIVE_FLUX_X,
      CONVECTIVE_FLUX_Y,
      CONVECTIVE_FLUX_Z,
      DIFFUSIVE_FLUX_X,
      DIFFUSIVE_FLUX_Y,
      DIFFUSIVE_FLUX_Z,
      SOURCE_TERM
    };

    /**
     * \todo currently we only allow one of each info type.  But there
     *       are cases where we may want multiple ones.  Example:
     *       diffusive terms in energy equation.  Expand this
     *       capability.
     */
    typedef std::map< FieldSelector, Expr::Tag > FieldTagInfo; //< Defines a map to hold information on ExpressionIDs for the RHS.
   

    /**
     *  \class Builder
     *  \author James C. Sutherland
     *  \date   June, 2010
     *
     *  \brief builder for ScalarRHS objecst.
     */
    class Builder : public Expr::ExpressionBuilder
    {
    public:

      /**
       *  \brief Constructs a builder for a ScalarRHS object.
       *
       *  \param fieldInfo the FieldTagInfo object that holds
       *         information for the various expressions that form the
       *         RHS.
       */
      Builder( const FieldTagInfo& fieldInfo );

      /**
       *  \brief Constructs a builder for a ScalarRHS object.
       *
       *  \param fieldInfo the FieldTagInfo object that holds
       *         information for the various expressions that form the
       *         RHS.
       *
       *  \param srcTags extra source terms to attach to this RHS.
       */
      Builder( const FieldTagInfo& fieldInfo,
               const std::vector<Expr::Tag>& srcTags );

      virtual ~Builder(){}

      virtual Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                           const Expr::ExpressionRegistry& reg ) const;
    protected:
      const FieldTagInfo info_;
      std::vector<Expr::Tag> srcT_;
    };

    virtual void evaluate();
    virtual void advertise_dependents( Expr::ExprDeps& exprDeps );
    virtual void bind_fields( const Expr::FieldManagerList& fml );
    virtual void bind_operators( const SpatialOps::OperatorDatabase& opDB );

  protected:

    const Expr::Tag convTagX_, convTagY_, convTagZ_;
    const Expr::Tag diffTagX_, diffTagY_, diffTagZ_;

    const bool haveConvection_, haveDiffusion_;
    const bool doXDir_, doYDir_, doZDir_;

    std::vector<Expr::Tag> srcTags_;

    const DivX* divOpX_;
    const DivY* divOpY_;
    const DivZ* divOpZ_;

    const XFluxT *xConvFlux_, *xDiffFlux_;
    const YFluxT *yConvFlux_, *yDiffFlux_;
    const ZFluxT *zConvFlux_, *zDiffFlux_;

    typedef std::vector<const FieldT*> SrcVec;
    SrcVec srcTerm_;

    void nullify_fields();

    static Expr::Tag resolve_field_tag( const typename ScalarRHS<FieldT>::FieldSelector field,
                                        const typename ScalarRHS<FieldT>::FieldTagInfo& info );
    
    ScalarRHS( const FieldTagInfo& fieldTags,
               const std::vector<Expr::Tag>& srcTags,
               const Expr::ExpressionID& id,
               const Expr::ExpressionRegistry& reg );
    
    virtual ~ScalarRHS();

  };


// ###################################################################
//
//                          Implementation
//
// ###################################################################

  
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
                                const Expr::ExpressionID& id,
                                const Expr::ExpressionRegistry& reg )
  : Expr::Expression<FieldT>( id, reg ),
  
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
  
  srcTags_( srcTags )
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
    
    divOpX_ = NULL;  divOpY_ = NULL;  divOpZ_ = NULL;
  }
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void ScalarRHS<FieldT>::bind_fields( const Expr::FieldManagerList& fml )
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
    
    for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
      if( isrc->context() != Expr::INVALID_CONTEXT )
        exprDeps.requires_expression( *isrc );
    }
  }
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void ScalarRHS<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
  {
    if( doXDir_ )  divOpX_ = opDB.retrieve_operator<DivX>();
    if( doYDir_ )  divOpY_ = opDB.retrieve_operator<DivY>();
    if( doZDir_ )  divOpZ_ = opDB.retrieve_operator<DivZ>();
  }
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void ScalarRHS<FieldT>::evaluate()
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
    
    typename SrcVec::const_iterator isrc;
    for( isrc=srcTerm_.begin(); isrc!=srcTerm_.end(); ++isrc ){
      rhs += **isrc;
    }
    
  }
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  ScalarRHS<FieldT>::Builder::Builder( const FieldTagInfo& fieldInfo,
                                       const std::vector<Expr::Tag>& sources )
  : info_( fieldInfo ),
  srcT_( sources )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  ScalarRHS<FieldT>::Builder::Builder( const FieldTagInfo& fieldInfo )
  : info_( fieldInfo )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  Expr::ExpressionBase*
  ScalarRHS<FieldT>::Builder::build( const Expr::ExpressionID& id,
                                     const Expr::ExpressionRegistry& reg ) const
  {
    return new ScalarRHS<FieldT>( info_, srcT_, id, reg );
  }
  //------------------------------------------------------------------
      
} // namespace Wasatch
#endif
