#ifndef SolnVarEst_Expr_h
#define SolnVarEst_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>


/**
 *  \ingroup WasatchExpressions
 *  \class  SolnVarEst
 *  \author Amir Biglari, Tony Saad
 *  \date	Mar, 2012
 *
 *  \brief Estimates the value of the solution variable in the next time step
 *         using a simple forward euler method. 
 *
 *  Note that this requires the solution variable, \f$\rho \phi\f$, and the 
 *  right hand side for the solution variable \f$ \frac{\partial \rho \phi}
 *  {\partial t} \f$ at the current RK stage. 
 */
template< typename FieldT >
class SolnVarEst 
  : public Expr::Expression<FieldT>
{  
  typedef SpatialOps::SingleValueField TimeField;

  DECLARE_FIELD(TimeField, dt_)
  DECLARE_FIELDS(FieldT, fOld_, rhs_)
  
  SolnVarEst( const Expr::Tag& solnVarOldTag,
              const Expr::Tag& solnVarRHSTag,
              const Expr::Tag& timeStepTag );
public:
  
  /**
   *  \brief Builder for the estimation of the solution variable in the next time step
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \brief Constructs a builder for the solution variable estimation
     *
     *  \param result the tag for the solution variable estimate
     *  \param solnVarOldTag a tag for solution variable at the current RK stage
     *  \param solnVarRHSTag a tag for the right hand side of solution variable
     *         transport equation at the current RK stage
     *  \param timeStepTag a tag for the time step at the current RK stage
     *
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& solnVarOldTag,
             const Expr::Tag& solnVarRHSTag,
             const Expr::Tag& timeStepTag );
    
    ~Builder(){}

    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag solnVarOldt_, solnVarRHSt_;
    const Expr::Tag tstpt_;

  };

  ~SolnVarEst();
  void evaluate();

};

#endif // SolnVarEst_Expr_h
