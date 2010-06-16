#ifndef ScalarRHS_h
#define ScalarRHS_h

#include <map>


//-- ExprLib Includes --//
#include <expression/ExprLib.h>


//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>


namespace Wasatch{

  /**
   *  @class ScalarRHS
   *  @author James C. Sutherland
   *
   *  @brief Support for a basic scalar transport equation involving
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
  class ScalarRHS
    : public Expr::Expression< Wasatch::ScalarVolField >
  {
  protected:

    typedef Wasatch::ScalarVolField           FieldT;
    typedef Wasatch::FaceTypes<FieldT>::XFace XFluxT;
    typedef Wasatch::FaceTypes<FieldT>::YFace YFluxT;
    typedef Wasatch::FaceTypes<FieldT>::ZFace ZFluxT;

    typedef Wasatch::OpTypes<FieldT>::DivX   DivX;
    typedef Wasatch::OpTypes<FieldT>::DivY   DivY;
    typedef Wasatch::OpTypes<FieldT>::DivZ   DivZ;

  public:

    enum FieldSelector{
      CONVECTIVE_FLUX_X,
      CONVECTIVE_FLUX_Y,
      CONVECTIVE_FLUX_Z,
      DIFFUSIVE_FLUX_X,
      DIFFUSIVE_FLUX_Y,
      DIFFUSIVE_FLUX_Z,
      SOURCE_TERM
    };

    // jcs currently we only allow one of each info type.  But there
    // are cases where we may want multiple ones.  Example: diffusive
    // terms in energy equation.
    typedef std::map< FieldSelector, Expr::Tag >  FieldTagInfo;

    class Builder : public Expr::ExpressionBuilder
    {
    public:

      Builder( const FieldTagInfo& fieldInfo );

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

    static Expr::Tag resolve_field_tag( const FieldSelector,
                                        const FieldTagInfo& fieldTags );

    ScalarRHS( const FieldTagInfo& fieldTags,
               const std::vector<Expr::Tag>& srcTags,
               const Expr::ExpressionID& id,
               const Expr::ExpressionRegistry& reg );

    virtual ~ScalarRHS();

  };

} // namespace Wasatch

#endif
