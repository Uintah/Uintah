#ifndef ParticleTemperatureRHS_coal_h
#define ParticleTemperatureRHS_coal_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

/**
 *   Author : Babak Goshayeshi, Josh McConnell
 *   Date   : April, 2018
 *   University of Utah - Institute for Clean and Secure Energy
 *
 *   Particle Temperature RHS, which depend on :
 *
 *   pMassTag           : particle mass
 *   pCpTag             : particle heat capacity ( constant pressure )
 *   evapRHSTag         : Moisture mass production rate in particle
 *   pTempTag           : particle temperature
 *   heatFromCharRxnsTag: heat from char oxidation and gasification
 *
 *
 *   Method -- Latent of Vaporization :
 *
 *   \Delta H_{v2}=\Delta H_{va}\left(\frac{1-T_{r2}}{1-T_{r1}}\right)^{n}
 *
 *   which :
 *
 *   \Delta H_{v2} : Latent heat of vaporization at the desired temperature
 *   \Delta H_{v2} : Latent heat of vaporization at the refrence temperature
 *   T_{r1}        : reduced temperature of refrence temperature
 *   T_{r1}        : reduced temperature of desired temperature
 *   n             : a common choice is 0.375 or 0.38
 */
namespace Coal {

template< typename FieldT >
class ParticleTemperatureRHS
 : public Expr::Expression<FieldT>
{
  const double alpha_;

  DECLARE_FIELDS( FieldT, pMass_, pCp_, evapRHS_, pTemp_, heatFromCharRxns_ )

  ParticleTemperatureRHS( const Expr::Tag& pMassTag,
                          const Expr::Tag& pCpTag,
                          const Expr::Tag& evapRHSTag,
                          const Expr::Tag& pTempTag,
                          const Expr::Tag& heatFromCharRxnsTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag pMassTag_,  pCpTag_ , evapRHSTag_,
                    pTempTag_, heatFromCharRxnsTag_;
  public:
    Builder( const Expr::Tag& pTempRHSTag,
             const Expr::Tag& pMassTag,
             const Expr::Tag& pCpTag,
             const Expr::Tag& evapRHSTag,
             const Expr::Tag& pTempTag,
             const Expr::Tag& heatFromCharRxnsTag )
    : Expr::ExpressionBuilder(pTempRHSTag),
      pMassTag_            ( pMassTag             ),
      pCpTag_              ( pCpTag               ),
      evapRHSTag_          ( evapRHSTag           ),
      pTempTag_            ( pTempTag             ),
      heatFromCharRxnsTag_ ( heatFromCharRxnsTag  )
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const{
      return new ParticleTemperatureRHS<FieldT>( pMassTag_, pCpTag_, evapRHSTag_,
                                                 pTempTag_, heatFromCharRxnsTag_ );
    }
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
ParticleTemperatureRHS<FieldT>::
ParticleTemperatureRHS( const Expr::Tag& pMassTag,
                        const Expr::Tag& pCpTag,
                        const Expr::Tag& evapRHSTag,
                        const Expr::Tag& pTempTag,
                        const Expr::Tag& heatFromCharRxnsTag )
  : Expr::Expression<FieldT>(),
    alpha_( Coal::absored_heat_fraction_particle() )
{
  this->set_gpu_runnable(true);

  pMass_            = this->template create_field_request<FieldT>( pMassTag            );
  pCp_              = this->template create_field_request<FieldT>( pCpTag              );
  evapRHS_          = this->template create_field_request<FieldT>( evapRHSTag          );
  pTemp_            = this->template create_field_request<FieldT>( pTempTag            );
  heatFromCharRxns_ = this->template create_field_request<FieldT>( heatFromCharRxnsTag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleTemperatureRHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& pMass    = pMass_           ->field_ref();
  const FieldT& pCp      = pCp_             ->field_ref();
  const FieldT& evapRHS  = evapRHS_         ->field_ref();
  const FieldT& pTemp    = pTemp_           ->field_ref();
  const FieldT& hSrcChar = heatFromCharRxns_->field_ref();

  // temperature rhs for vaporization
  // @ T = 300 K ,Heat of Vaporization = 2438 kj/kg, Tcritical = 647.096 K
  const double Tc = 647.096;

  // temperature rhs for char oxidation
  result <<=
  (
      cond( pTemp < Tc, 2438.E3 * pow( (1. - pTemp/Tc)/(1. - 300/Tc), 0.375 ) * evapRHS )
          ( 0. )  // vaporization
      -alpha_ * hSrcChar   // char rxns. hSrcChar is negative when exothermic, hence the "-"
  )
  / (pMass * pCp);
}

//--------------------------------------------------------------------

} // namespace coal
/*
 [1] The Combustion Rates of Coal Chars : A Review, L. W. Smith, symposium on combustion 19, 1982, pp 1045-1065
 [2] Watanabe, H, and M Otaka. Numerical simulation of coal gasification in entrained flow coal gasifier
     Fuel 85, no. 12-13 (September 2006): 1935-1943.
      http://linkinghub.elsevier.com/retrieve/pii/S0016236106000548.
 */
#endif // ParticleTemperatureRHS_CHAR_h
