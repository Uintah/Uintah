#ifndef Drhodt_Expr_h
#define Drhodt_Expr_h

#include <map>

//-- ExprLib Includes --//
#include <expression/Expression.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>


/**
 *  \ingroup WasatchExpressions
 *  \class  Drhodt
 *  \author Amir Biglari, Tony Saad
 *  \date	Octtober, 2013
 *
 *  \brief Reproduces the model that we use in the pressure source term to calculate
 *         \f$ \nabla\cdot(\rho u)^{n+1} \f$ which is \f$ (1-\alpha) \nabla\cdot 
 *         (\rho^{n+1}u^{n+1}) - \alpha (\frac{\partial\rho}{\partial t})^{n+1} \f$.
 *         In these calculations drhodt at n+1 is calculated using a central 
 *         differencing in time, u at n+1 is estimated using the weak form of the 
 *         momentum equation with a forward Euler method and density is obtained  
 *         from the estimations of the scalars in their conservative form obtained 
 *         with a forward Euler method as well.
 *
 */
class Drhodt : public Expr::Expression<SVolField>
{  
  
  typedef SpatialOps::structured::FaceTypes<SVolField> FaceTypes;
  typedef FaceTypes::XFace XFace; ///< The type of field for the x-face of SVolField.
  typedef FaceTypes::YFace YFace; ///< The type of field for the y-face of SVolField.
  typedef FaceTypes::ZFace ZFace; ///< The type of field for the z-face of SVolField.

  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, XFace >::type XFaceInterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, YFace >::type YFaceInterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, ZFace >::type ZFaceInterpT;
  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, XFace >::type Scalar2XFInterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, YFace >::type Scalar2YFInterpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, ZFace >::type Scalar2ZFInterpT;
  
  typedef SpatialOps::structured::BasicOpTypes<SVolField> OpTypes;
  typedef OpTypes::DivX DivXT; ///< Divergence operator (surface integral) in the x-direction
  typedef OpTypes::DivY DivYT; ///< Divergence operator (surface integral) in the y-direction
  typedef OpTypes::DivZ DivZT; ///< Divergence operator (surface integral) in the z-direction
  
  const XVolField *uStar_;
  const YVolField *vStar_;
  const ZVolField *wStar_;
  const SVolField *dens_, *densStar_, *dens2Star_;
  
  typedef SpatialOps::structured::SingleValueField TimeField;
  const bool doX_, doY_, doZ_;
  const TimeField* timestep_;

  const Expr::Tag xVelStart_, yVelStart_, zVelStart_, denst_, densStart_, dens2Start_, timestept_;
  
  const DivXT* divXOp_;
  const DivYT* divYOp_;
  const DivZT* divZOp_;
  const XFaceInterpT* xFInterpOp_;
  const YFaceInterpT* yFInterpOp_;
  const ZFaceInterpT* zFInterpOp_;
  const Scalar2XFInterpT* s2XFInterpOp_;
  const Scalar2YFInterpT* s2YFInterpOp_;
  const Scalar2ZFInterpT* s2ZFInterpOp_;
  
  
  Drhodt( const Expr::TagList& velStarTags,
             const Expr::Tag densTag,
             const Expr::Tag densStarTag,
             const Expr::Tag dens2StarTag,
             const Expr::Tag timestepTag);
public:
  
  /**
   *  \brief Builder for drhodt at time step n+1 which is modeled in pressure source 
   *         term expression as well (i.e. the second derivaive of density with
   *         respect to time)
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \brief Constructs a builder for source term of the pressure
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
     *  \param timestepTag a tag to hold the timestep value.
     */
    Builder( const Expr::Tag& result,
             const Expr::TagList& velStarTags,
             const Expr::Tag densTag,
             const Expr::Tag densStarTag,
             const Expr::Tag dens2StarTag,
             const Expr::Tag timestepTag );
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::TagList velStarTs_;
    const Expr::Tag denst_, densStart_, dens2Start_, tstpt_;
  };
  
  ~Drhodt();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

#endif // Drhodt_Expr_h
