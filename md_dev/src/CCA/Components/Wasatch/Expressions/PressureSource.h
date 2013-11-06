#ifndef PressureSource_Expr_h
#define PressureSource_Expr_h

#include <map>

//-- ExprLib Includes --//
#include <expression/Expression.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>


/**
 *  \ingroup WasatchExpressions
 *  \class  PressureSource
 *  \author Amir Biglari
 *  \date	July, 2011
 *
 *  \brief Calculates the source term for the pressure equation ( i.e. the second
 *         derivaive of density with respect to time) of the form \f$ \frac{
 *         \nabla.(\rho u)^n + (\alpha (\frac{\partial \rho}{\partial t})^{n+1} 
 *         - (1 - \alpha) \nabla.(\rho u)^{n+1}) }{\Delta t} = \frac{\partial^2
 *         \rho}{\partial t^2} \f$ for variable density low-Mach problems, where \f$\alpha\f$ 
 *         is a weighting factor applied on the continuity equation and added to the 
 *         equations. This requires knowledge of the momentum field and, the velocity  
 *         vactor and the density field in future time steps.
 *         OR
 *         of the form \f$ \frac{ \rho \nabla.u^n }{\Delta t} = \frac{\partial^2
 *         \rho}{\partial t^2} \f$ which requires knowledge of a dilitation field.
 *  
 *  Note that here the dilitation at time step n+1 is enforced to be 0, but at the
 *       current time step is allowed not to be zero. By retaining this term, we can
 *       remain consistent even when initial conditions do not satisfy the governing
 *       equations.
 *
 */
class PressureSource : public Expr::Expression<SVolField>
{  
  
  typedef SpatialOps::structured::FaceTypes<SVolField> FaceTypes;
  typedef FaceTypes::XFace XFace; ///< The type of field for the x-face of SVolField.
  typedef FaceTypes::YFace YFace; ///< The type of field for the y-face of SVolField.
  typedef FaceTypes::ZFace ZFace; ///< The type of field for the z-face of SVolField.

  typedef SpatialOps::structured::SingleValueField TimeField;

  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, XVolField >::type S2XInterpOpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, YVolField >::type S2YInterpOpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, ZVolField >::type S2ZInterpOpT;
  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SVolField >::type GradXT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SVolField >::type GradYT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SVolField >::type GradZT;
  
  const XVolField *xMom_, *uStar_;
  const YVolField *yMom_, *vStar_;
  const ZVolField *zMom_, *wStar_;
  const SVolField *dens_, *densStar_, *dens2Star_;
  const SVolField *dil_;
  const TimeField* timestep_;
  
  const bool isConstDensity_, doX_, doY_, doZ_, is3d_;
  
  const Expr::Tag xMomt_, yMomt_, zMomt_;
  const Expr::Tag xVelStart_, yVelStart_, zVelStart_, denst_, densStart_, dens2Start_, dilt_, timestept_;
  
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;
  const S2XInterpOpT* s2XInterpOp_;
  const S2YInterpOpT* s2YInterpOp_;
  const S2ZInterpOpT* s2ZInterpOp_;  
  
  PressureSource( const Expr::TagList& momTags,
                  const Expr::TagList& velStarTags,
                  const bool isConstDensity,
                  const Expr::Tag densTag,
                  const Expr::Tag densStarTag,
                  const Expr::Tag dens2StarTag );
public:
  
  /**
   *  \brief Builder for the source term in the pressure poisson equation 
   *         (i.e. the second derivaive of density with respect to time)
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \brief Constructs a builder for source term of the pressure
     *
     *  \param the momTags a list tag which holds the tags for momentum in
     *         all directions
     *
     *  \param the velStarTags a list tag which holds the tags for velocity at 
     *         the time stage "*" in all directions     
     *
     *  \param densTag a tag to hold density in constant density cases, which is 
     *         needed to obtain drhodt 
     *
     *  \param densStarTag a tag for estimation of density at the time stage "*"
     *         which is needed to obtain momentum at that stage.
     *
     *  \param dens2StarTag a tag for estimation of density at the time stage "**"
     *         which is needed to calculate drhodt 
     *
     *  \param dilTag a tag to hold dilatation term in constant density cases.
     *
     *  \param timestepTag a tag to hold the timestep value.
     */
    Builder( const Expr::TagList& results,
             const Expr::TagList& momTags,
             const Expr::TagList& velStarTags,
             const bool isConstDensity,
             const Expr::Tag densTag,
             const Expr::Tag densStarTag,
             const Expr::Tag dens2StarTag );
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const;
    
  private:
    const bool isConstDens_;
    const Expr::TagList momTs_, velStarTs_;
    const Expr::Tag denst_, densStart_, dens2Start_;
  };
  
  ~PressureSource();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

#endif // PressureSource_Expr_h
