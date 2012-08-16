
#ifndef PrecipitateEffectiveViscosity_Expr_h
#define PrecipitateEffectiveViscosity_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitateEffectiveViscosity
 *  \author Alex Abboud	 
 *  \date June 2012
 *  \brief Modifies Einstein Viscosity -a first order taylor series expansion derived from kinetic theory
 *  to account for shear thinning effect
 *  \f$ \mu_r = \mu / \mu_0 = ( 1 + 2.5 \lambda \phi ) |S|^n \f$
 *  Best used for particle volume fractionss < 0.10
 */
template< typename FieldT >
class PrecipitateEffectiveViscosity
: public Expr::Expression<FieldT>
{
  
  const Expr::Tag volumeFractionTag_;  //Tag for particle volume fraction
  const Expr::Tag strainMagnitudeTag_; //Tag for strain magnitude
  const double corrFac_; 							 //correction factor \lambda
  const double baseViscosity_;         // \mu_0
  const double power_;                 //power law exponent (n)
  const double minStrain_;             //cutoff so that vscosity does not go too high
  const FieldT* volumeFraction_;     
  const FieldT* strainMagnitude_;
  
  PrecipitateEffectiveViscosity( const Expr::Tag& volumeFractionTag,
                                 const Expr::Tag& strainMagnitudeTag,
                                 const double corrFac,
                                 const double baseViscosity,
                                 const double power,
                                 const double minStrain);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& volumeFractionTag,
             const Expr::Tag& strainMagnitudeTag,
             const double corrFac,
             const double baseViscosity,
             const double power,
             const double minStrain)
    : ExpressionBuilder(result),
    volumefractiont_(volumeFractionTag),
    strainmagnitudet_(strainMagnitudeTag),
    corrfac_(corrFac),
    baseviscosity_(baseViscosity),
    power_(power),
    minstrain_(minStrain)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new PrecipitateEffectiveViscosity<FieldT>( volumefractiont_, strainmagnitudet_, corrfac_, baseviscosity_, power_, minstrain_ );
    }
    
  private:
    const Expr::Tag volumefractiont_;
    const Expr::Tag strainmagnitudet_;
    const double corrfac_;
    const double baseviscosity_;
    const double power_;
    const double minstrain_;
  };
  
  ~PrecipitateEffectiveViscosity();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
PrecipitateEffectiveViscosity<FieldT>::
PrecipitateEffectiveViscosity( const Expr::Tag& volumeFractionTag,
                               const Expr::Tag& strainMagnitudeTag,
                               const double corrFac,
                               const double baseViscosity,
                               const double power,
                               const double minStrain)
: Expr::Expression<FieldT>(),
volumeFractionTag_(volumeFractionTag),
strainMagnitudeTag_(strainMagnitudeTag),
corrFac_(corrFac),
baseViscosity_(baseViscosity),
power_(power),
minStrain_(minStrain)
{}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitateEffectiveViscosity<FieldT>::
~PrecipitateEffectiveViscosity()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitateEffectiveViscosity<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( volumeFractionTag_ );
  exprDeps.requires_expression( strainMagnitudeTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitateEffectiveViscosity<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  //const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  volumeFraction_ = &fm.field_ref( volumeFractionTag_ );
  strainMagnitude_ = &fm.field_ref( strainMagnitudeTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitateEffectiveViscosity<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitateEffectiveViscosity<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  
  FieldT& result = this->value();
  //multiplying |S| by 2.0 & raising to 1/2, since wasatch turb model returns SijSij
  result <<= cond( *volumeFraction_ < 1e-10 , baseViscosity_) 
                 ( 1.0 > (1 + 2.5 * corrFac_ * *volumeFraction_ ) * pow(2.0 * *strainMagnitude_, power_/2 ), baseViscosity_)
                 ( sqrt(2.0* *strainMagnitude_) < minStrain_ && 1.0 > (1 + 2.5 * corrFac_ * *volumeFraction_ ) * pow( minStrain_ , power_ ),  baseViscosity_)
                 ( sqrt(2.0* *strainMagnitude_) < minStrain_  , ( 1 + 2.5 * corrFac_ * *volumeFraction_ ) * pow( minStrain_ , power_ ) * baseViscosity_ ) 
                 ( (1 + 2.5 * corrFac_ * *volumeFraction_ ) * pow(2.0 * *strainMagnitude_, power_/2 ) * baseViscosity_ );

}

#endif

