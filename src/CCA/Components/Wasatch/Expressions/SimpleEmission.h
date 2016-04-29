#ifndef SimpleEmission_Expr_h
#define SimpleEmission_Expr_h

#include <expression/Expression.h>

/**
 *  \class SimpleEmission
 *  \author James C. Sutherland
 *
 *  \f[ \nabla \cdot \vec{q} = \sigma \epsilon (T^4 - T_\infty^4) \f]
 */
template< typename FieldT >
class SimpleEmission
 : public Expr::Expression<FieldT>
{
  const double envTempValue_;
  const bool hasAbsCoef_, hasConstEnvTemp_;
  
  DECLARE_FIELDS( FieldT, temperature_, envTemp_, absCoef_ )

  SimpleEmission( const Expr::Tag& temperatureTag,
                  const Expr::Tag& envTempTag,
                  const double envTemp,
                  const Expr::Tag& absCoefTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a SimpleEmission expression
     *  @param resultTag the tag for the value that this expression computes
     *  @param temperatureTag the gas temperature
     *  @param envTempTag the radiative temperature of the surroundings
     *  @param absCoefTag the absorption coefficient (optional)
     */
    Builder( const Expr::Tag divQTag,
             const Expr::Tag temperatureTag,
             const Expr::Tag envTempTag,
             const Expr::Tag absCoefTag );

    /**
     *  @brief Build a SimpleEmission expression
     *  @param resultTag the tag for the value that this expression computes
     *  @param temperatureTag the gas temperature
     *  @param envTempTag the (constant) radiative temperature of the surroundings
     *  @param absCoefTag the absorption coefficient (optional)
     */
    Builder( const Expr::Tag divQTag,
             const Expr::Tag temperatureTag,
             const double envTemp,
             const Expr::Tag absCoefTag );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag temperatureTag_, envTempTag_, absCoefTag_;
    const double envTemp_;
  };

  void evaluate();
};


#endif // SimpleEmission_Expr_h
