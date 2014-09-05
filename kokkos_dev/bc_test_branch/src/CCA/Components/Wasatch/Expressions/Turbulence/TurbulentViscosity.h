#ifndef TurbulentViscosity_Expr_h
#define TurbulentViscosity_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include "TurbulenceParameters.h"

#include <expression/Expression.h>

/**
 *  \class TurbulentViscosity
 *  \author Tony Saad, Amir Biglari
 *  \date   June, 2012. (Originally created: Jan, 2012).
 *  \ingroup Expressions
 *  \brief given a strain tensor magnitude, \f$|\tilde{S}| = ( 2\tilde{S_{kl}}\tilde{S_{kl}} )^{1/2}\f$, Smagorinsky constant, \f$C\f$, filtered density, \f$\bar{\rho}\f$, Kolmogorov length scale, \f$\eta\f$, and molecular viscosity, \f$\mu\f$,this calculates mixed viscosity, \f$\mu_{T} + \mu = C \bar{\rho} \Delta^2 |\tilde{S}| + \mu\f$.
 *
 *   Note: It is currently assumed that mixed viscosity is a "SVolField" type.  
 *         Therefore, variables should be interpolated into "SVolField" at the end. 
 *         Note that, there are two different cases of builders, one for constant 
 *         Smagorinsky constant, and the other one is for variable Smagorinsky 
 *         constant, e.g. when we have dynamic model for this constant.
 *
 *   Note: It is currently assumed that filtered density and molecular viscosity are 
 *         "SVolField" type. Therefore, there is no need to interpolate it.
 *
 */

class TurbulentViscosity
 : public Expr::Expression<SVolField>
{
  
  const bool isConstSmag_;
  Wasatch::TurbulenceParameters turbulenceParameters_;
  const Expr::Tag strTsrSqTag_, waleTsrMagTag_, vremanTsrMagTag_, dynCoefTag_, rhoTag_;

  // gradient operators are only here to extract spacing information out of them
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SVolField >::type GradXT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SVolField >::type GradYT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SVolField >::type GradZT;
  
  typedef Wasatch::OpTypes<SVolField>::BoundaryExtrapolant ExOpT;
  
  const GradXT*  gradXOp_;            ///< x-component of the gradient operator
  const GradYT*  gradYOp_;            ///< y-component of the gradient operator  
  const GradZT*  gradZOp_;            ///< z-component of the gradient operator
  ExOpT*   exOp_;

  const SVolField *dynCoef_, *rho_, *strTsrSq_, *waleTsrMag_, *vremanTsrMag_;

  TurbulentViscosity( const Expr::Tag rhoTag,
                      const Expr::Tag strTsrSqTag,
                      const Expr::Tag waleTsrMagTag,
                      const Expr::Tag vremanTsrMagTag,
                      const Expr::Tag dynamicSmagCoefTag,
                      const Wasatch::TurbulenceParameters turbParams);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a turbulent viscosity given expressions for
     *         \f$\rho\f$, \f$u_1\f$, \f$u_2\f$ and a constant value for
     *         Smagorinsky Constant, \f$C\f$.
     *
     *  \param rhoTag the Expr::Tag for the density.
     *
     *  \param strTsrSqTag the Expr::Tag holding the strain tensor magnitude 
     *
     *  \param smag the value (constant in space and time) for the
     *         Smagorinsky Constant.
     *
     *  \param kolmogScale the value (constant in space and time"is it?!") for the
     *         Kolmogrov length scale (should be define in input file)
     *
     *  \param viscTag the Expr::Tag for the molecular viscosity
     *
     */    
    Builder( const Expr::Tag& result,
             const Expr::Tag rhoTag,
             const Expr::Tag strTsrSqTag,
             const Expr::Tag waleTsrMagTag,
             const Expr::Tag vremanTsrMagTag,
             const Expr::Tag dynamicSmagCoefTag,
             const Wasatch::TurbulenceParameters turbParams )
      : ExpressionBuilder(result),
        isConstSmag_         ( true                ),
        turbulenceParameters_( turbParams          ),
        rhot_                ( rhoTag              ),
        strTsrSqt_           ( strTsrSqTag         ),
        waleTsrMagt_         ( waleTsrMagTag       ),
        vremanTsrMagt_       ( vremanTsrMagTag     ),
        dynCoeft_            ( dynamicSmagCoefTag )
    {}
    
    Expr::ExpressionBase* build() const
    {
      return new TurbulentViscosity( rhot_, strTsrSqt_, waleTsrMagt_, vremanTsrMagt_, dynCoeft_, turbulenceParameters_ );
    }    
  private:
    const bool isConstSmag_;
    const Wasatch::TurbulenceParameters turbulenceParameters_;
    const Expr::Tag rhot_, strTsrSqt_, waleTsrMagt_, vremanTsrMagt_, dynCoeft_;
  };

  ~TurbulentViscosity();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // TurbulentViscosity_Expr_h
