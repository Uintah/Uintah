#ifndef ScalarRHS_h
#define ScalarRHS_h

#include <map>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>


/**
 *  \ingroup 	Expressions
 *  \class 	ScalarRHS
 *  \author 	James C. Sutherland
 *
 *  \brief Support for a basic scalar transport equation involving
 *         any/all of advection, diffusion and reaction.
 *
 *  \tparam FieldT - the type of field for the RHS.
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

  typedef typename SpatialOps::structured::FaceTypes<FieldT> FaceTypes;
  typedef typename FaceTypes::XFace XFluxT; ///< The type of field for the x-face variables.
  typedef typename FaceTypes::YFace YFluxT; ///< The type of field for the y-face variables.
  typedef typename FaceTypes::ZFace ZFluxT; ///< The type of field for the z-face variables.

  typedef typename SpatialOps::structured::BasicOpTypes<FieldT> OpTypes;
  typedef typename OpTypes::DivX   DivX; ///< Divergence operator (surface integral) in the x-direction
  typedef typename OpTypes::DivY   DivY; ///< Divergence operator (surface integral) in the y-direction
  typedef typename OpTypes::DivZ   DivZ; ///< Divergence operator (surface integral) in the z-direction

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

#endif
