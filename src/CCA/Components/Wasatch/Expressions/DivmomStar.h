#ifndef DivmomStar_Expr_h
#define DivmomStar_Expr_h

#include <map>

//-- ExprLib Includes --//
#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/VardenParameters.h>

/**
 *  \ingroup WasatchExpressions
 *  \class  DivmomStar
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
class DivmomStar : public Expr::Expression<SVolField>
{  
  
  typedef SpatialOps::FaceTypes<SVolField> FaceTypes;
  typedef FaceTypes::XFace XFace; ///< The type of field for the x-face of SVolField.
  typedef FaceTypes::YFace YFace; ///< The type of field for the y-face of SVolField.
  typedef FaceTypes::ZFace ZFace; ///< The type of field for the z-face of SVolField.

  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, XVolField >::type S2XInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, YVolField >::type S2YInterpOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, ZVolField >::type S2ZInterpOpT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SVolField >::type GradXT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SVolField >::type GradYT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SVolField >::type GradZT;
  
  DECLARE_FIELD(XVolField, uStar_);
  DECLARE_FIELD(YVolField, vStar_);
  DECLARE_FIELD(ZVolField, wStar_);
  DECLARE_FIELD(SVolField, densStar_);
  
  const bool doX_, doY_, doZ_, is3d_;
  
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;
  const S2XInterpOpT* s2XInterpOp_;
  const S2YInterpOpT* s2YInterpOp_;
  const S2ZInterpOpT* s2ZInterpOp_;  
  
  DivmomStar( const Expr::TagList& velStarTags,
              const Expr::Tag densStarTag );
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
     *  \param result the divergence of the momentum predictor
     *  \param velStarTags a list tag which holds the tags for velocity at
     *         the time stage "*" in all directions     
     *  \param densStarTag a tag for estimation of density at the time stage "*"
     *         which is needed to obtain momentum at that stage.
     */
    Builder( const Expr::Tag& result,
             const Expr::TagList& velStarTags,
             const Expr::Tag densStarTag );
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::TagList velStarTs_;
    const Expr::Tag densStart_;
  };
  
  ~DivmomStar();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

#endif // DivmomStar_Expr_h
