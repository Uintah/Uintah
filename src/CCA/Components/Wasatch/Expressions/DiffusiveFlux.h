#ifndef DiffusiveFlux_Expr_h
#define DiffusiveFlux_Expr_h

//-- ExprLib includes --//
#include <expression/Expr_Expression.h>

//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggeredOperatorTypes.h>


/**
 *  \ingroup Expressions
 *  \class  DiffusiveFlux
 *  \author James C. Sutherland
 *  \date	June, 2010
 *  \modifier Amir Biglari
 *  \date July,2011
 *
 *  \brief Calculates a simple diffusive flux of the form
 *         \f$ J_i = -\rho \Gamma \frac{\partial \phi}{\partial x_i} \f$
 *         where \f$i=1,2,3\f$ is the coordinate direction.
 *         This requires knowledge of a the velocity field.
 *
 *  \tparam ScalarT the type for the scalar primary variable.
 *
 *  \tparam FluxT the type for the diffusive flux which is actually the scalarT faces.
 *
 *  Note that this requires the diffusion coefficient, \f$\Gamma\f$,
 *  to be evaluated at the same location as \f$J_i\f$ and
 *  \f$\frac{\partial \phi}{\partial x_i}\f$.
 */
template< typename ScalarT, typename FluxT >
class DiffusiveFlux
  : public Expr::Expression< FluxT >
{
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,  FluxT>::type  GradT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FluxT>::type  DensityInterpT;

  const bool  isConstCoef_;
  const Expr::Tag phiTag_, coefTag_, rhoTag_;
  const double coefVal_;

  const GradT*          gradOp_;
  const DensityInterpT* densityInterpOp_;

  const ScalarT*   phi_;
  const SVolField* rho_;
  const FluxT*     coef_;

  DiffusiveFlux( const Expr::Tag rhoTag,
                 const Expr::Tag phiTag,
                 const Expr::Tag coefTag,
                 const Expr::ExpressionID& id,
                 const Expr::ExpressionRegistry& reg );

  DiffusiveFlux( const Expr::Tag rhoTag,
                 const Expr::Tag phiTag,
                 const double coefTag,
                 const Expr::ExpressionID& id,
                 const Expr::ExpressionRegistry& reg );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = -\rho \Gamma \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the
   *         same location as \f$J\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a diffusive flux given expressions for
     *         \f$\phi\f$ and \f$\Gamma\f$
     *
     *  \param phiTag  the Expr::Tag for the scalar field
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the flux field).
     *
     *  \param rhoTag the Expr::Tag for the density which will be interpolated to FluxT field.
     */
    Builder( const Expr::Tag phiTag,
             const Expr::Tag coefTag,
             const Expr::Tag rhoTag = Expr::Tag() )
      : isConstCoef_( false ),
        phit_(phiTag),
        coeft_(coefTag),
        rhot_(rhoTag),
        coef_(0.0)
    {}

    /**
     *  \brief Construct a diffusive flux given an expression for
     *         \f$\phi\f$ and a constant value for \f$\Gamma\f$.
     *
     *  \param phiTag  the Expr::Tag for the scalar field
     *
     *  \param coef the value (constant in space and time) for the
     *         diffusion coefficient.
     *
     *  \param rhoTag the Expr::Tag for the density which will be interpolated to FluxT field.
     */
    Builder( const Expr::Tag phiTag,
             const double coef,
             const Expr::Tag rhoTag = Expr::Tag() )
      : isConstCoef_( true ),
        rhot_(rhoTag),
        phit_(phiTag),
        coef_(coef)
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const
    {
      if( isConstCoef_ ) return new DiffusiveFlux<ScalarT, FluxT>( rhot_, phit_, coef_, id, reg );
      else               return new DiffusiveFlux<ScalarT, FluxT>( rhot_, phit_, coeft_, id, reg );
    }
  private:
    const bool isConstCoef_;
    const Expr::Tag phit_,coeft_,rhot_;
    const double coef_;
  };

  ~DiffusiveFlux();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};




/**
 *  \ingroup WasatchExpressions
 *  \class  DiffusiveFlux2
 *  \author James C. Sutherland
 *  \date   June, 2010
 *  \modifier Amir Biglari
 *  \date July,2011
 *
 *  \brief Calculates a generic diffusive flux, \f$J = -\rho \Gamma
 *         \frac{\partial \phi}{\partial x}\f$, where \f$\Gamma\f$ is
 *         located at the same location as \f$\phi\f$.
 *
 *  \tparam ScalarT the type for the scalar primary variable.
 *
 *  \tparam FluxT the type for the diffusive flux which is actually the scalarT faces.
 *
 *  \tparam InterpT The type of operator used in interpolating
 *       \f$\Gamma\f$ from the location of \f$\phi\f$ to the location
 *       of \f$\frac{\partial \phi}{\partial x}\f$
 */
template< typename ScalarT,
          typename FluxT >
class DiffusiveFlux2
  : public Expr::Expression< FluxT >
{
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,  FluxT>::type  GradT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,ScalarT,  FluxT>::type  InterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FluxT>::type  DensityInterpT;

  const Expr::Tag phiTag_, coefTag_, rhoTag_;

  const GradT* gradOp_;
  const InterpT* interpOp_;
  const DensityInterpT* densityInterpOp_;

  const ScalarT* phi_;
  const SVolField* rho_;
  const ScalarT* coef_;

  DiffusiveFlux2( const Expr::Tag rhoTag,
                  const Expr::Tag phiTag,
                  const Expr::Tag coefTag,
                  const Expr::ExpressionID& id,
                  const Expr::ExpressionRegistry& reg );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = - \rho \Gamma \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the
   *         same location as \f$\phi\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a DiffusiveFlux2::Builder object for
     *         registration with an Expr::ExpressionFactory.
     *
     *  \param phiTag the Expr::Tag for the scalar field.
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the scalar field).
     *
     *  \param rhoTag the Expr::Tag for the density which will be interpolated to FluxT field.
     */
    Builder( const Expr::Tag phiTag,
             const Expr::Tag coefTag,
             const Expr::Tag rhoTag = Expr::Tag() )
    : phit_(phiTag),
      coeft_( coefTag ),
      rhot_(rhoTag)
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const{
      return new DiffusiveFlux2<ScalarT,FluxT>( rhot_, phit_, coeft_, id, reg );
    }
  private:
    const Expr::Tag phit_,coeft_,rhot_;
  };

  ~DiffusiveFlux2();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // DiffusiveFlux_Expr_h
