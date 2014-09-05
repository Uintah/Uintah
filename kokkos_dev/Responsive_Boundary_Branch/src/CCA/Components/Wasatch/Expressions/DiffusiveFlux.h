#ifndef DiffusiveFlux_Expr_h
#define DiffusiveFlux_Expr_h

#include <expression/Expr_Expression.h>

/**
 *  \ingroup Expressions
 *  \class  DiffusiveFlux
 *  \author James C. Sutherland
 *  \date	June, 2010
 *
 *  \brief Calculates a simple diffusive flux of the form
 *         \f$ J_i = -\Gamma \frac{\partial \phi}{\partial x_i} \f$ 
 *         where \f$i=1,2,3\f$ is the coordinate direction.
 *         This requires knowledge of a the velocity field.
 *
 *  \tparam GradT the type for the gradient operator.
 *
 *  Note that this requires the diffusion coefficient, \f$\Gamma\f$,
 *  to be evaluated at the same location as \f$J_i\f$ and 
 *  \f$\frac{\partial \phi}{\partial x_i}\f$.
 */
template< typename GradT >
class DiffusiveFlux
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType FluxT;
  typedef typename GradT::SrcFieldType  ScalarT;

  const bool isConstCoef_;
  const Expr::Tag phiTag_, coefTag_;
  const double coefVal_;

  const GradT* gradOp_;
  const ScalarT* phi_;
  const FluxT* coef_;

  DiffusiveFlux( const Expr::Tag phiTag,
                 const Expr::Tag coefTag,
                 const Expr::ExpressionID& id,
                 const Expr::ExpressionRegistry& reg );

  DiffusiveFlux( const Expr::Tag phiTag,
                 const double coefTag,
                 const Expr::ExpressionID& id,
                 const Expr::ExpressionRegistry& reg );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = -\Gamma \frac{\partial
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
     */
    Builder( const Expr::Tag phiTag, const Expr::Tag coefTag )
      : isConstCoef_( false ),
        phit_(phiTag),
        coeft_(coefTag),
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
     */
    Builder( const Expr::Tag phiTag, const double coef )
      : isConstCoef_( true ),
        phit_(phiTag),
        coef_(coef)
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const
    {
      if( isConstCoef_ ) return new DiffusiveFlux<GradT>( phit_, coef_,  id, reg );
      else               return new DiffusiveFlux<GradT>( phit_, coeft_, id, reg );
    }
  private:
        const bool isConstCoef_;
    const Expr::Tag phit_,coeft_;
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
 *
 *  \brief Calculates a generic diffusive flux, \f$J = -\Gamma
 *         \frac{\partial \phi}{\partial x}\f$, where \f$\Gamma\f$ is
 *         located at the same location as \f$\phi\f$.
 *
 *  \tparam GradT The type of operator used in forming
 *       \f$\frac{\partial \phi}{\partial x}\f$
 *
 *  \tparam InterpT The type of operator used in interpolating
 *       \f$\Gamma\f$ from the location of \f$\phi\f$ to the location
 *       of \f$\frac{\partial \phi}{\partial x}\f$
 */
template< typename GradT,
          typename InterpT >
class DiffusiveFlux2
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType FluxT;
  typedef typename GradT::SrcFieldType  ScalarT;

  const Expr::Tag phiTag_, coefTag_;

  const GradT* gradOp_;
  const InterpT* interpOp_;

  const ScalarT* phi_;
  const ScalarT* coef_;

  DiffusiveFlux2( const Expr::Tag phiTag,
                  const Expr::Tag coefTag,
                  const Expr::ExpressionID& id,
                  const Expr::ExpressionRegistry& reg );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = -\Gamma \frac{\partial
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
     */
    Builder( const Expr::Tag phiTag, const Expr::Tag coefTag )
      : phit_(phiTag), coeft_( coefTag )
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const    {
      return new DiffusiveFlux2<GradT,InterpT>( phit_, coeft_, id, reg );
    }
  private:
    const Expr::Tag phit_,coeft_;
  };

  ~DiffusiveFlux2();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // DiffusiveFlux_Expr_h
