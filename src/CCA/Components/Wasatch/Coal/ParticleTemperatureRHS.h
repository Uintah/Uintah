#ifndef ParticleTemperatureRHS_coal_h
#define ParticleTemperatureRHS_coal_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

/**
 *   Author : Babak Goshayeshi (www.bgoshayeshi.com)
 *   Date   : Feb 11 2011
 *   University of Utah - Institute for Clean and Secure Energy
 *
 *   Particle Temperature RHS, which depend on :
 *
 *   partmast   : particle mass
 *   partCpt    : particle heat capacity ( constant pressure )
 *   moistrhst  : Moisture mass production rate in particle
 *   oxidationt : Char Oxidaiot RHS
 *   co2gasift  : Char consumption rate due to CO2 gasification reaction
 *   h2ogasift  : Char consumption rate due to h2o gasification reaction
 *   co2coratiot: CO2 / CO - From char oxidation reaction
 *   tempPt     : particle temperature
 *   o2rhst     : Oxygen consumption rate
 *   inttempGt  : interpolated gas temperature to particle filed
 *   intpresst  : interpolated gas pressure to particle field
 *   co2corhst  : co2 and co consumption rate (in gas phase)
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

  DECLARE_FIELDS( FieldT, partmas_, partCp_, moistrhs_, oxidation_, co2gasif_ )
  DECLARE_FIELDS( FieldT, h2ogasif_, co2coratio_, tempP_, inttempG_ )
	

  ParticleTemperatureRHS( const Expr::Tag& partmast,
                          const Expr::Tag& partCpt,
                          const Expr::Tag& moistrhst,
                          const Expr::Tag& oxidationt,
                          const Expr::Tag& co2gasift,
                          const Expr::Tag& h2ogasift,
                          const Expr::Tag& co2coratiot,
                          const Expr::Tag& tempPt,
                          const Expr::Tag& inttempGt );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag partmast_,  partCpt_ , moistrhst_, oxidationt_, co2gasift_,
                    h2ogasift_, co2coratiot_, inttempGt_, tempPt_;
  public:
    Builder( const Expr::Tag& ptrhs,
             const Expr::Tag& partmast,
             const Expr::Tag& partCpt,
             const Expr::Tag& moistrhst,
             const Expr::Tag& oxidaitont,
             const Expr::Tag& co2gasift,
             const Expr::Tag& h2ogasift,
             const Expr::Tag& co2coratiot,
             const Expr::Tag& tempPt,
             const Expr::Tag& inttempGt )
    : Expr::ExpressionBuilder(ptrhs),
      partmast_   ( partmast   ),
      partCpt_    ( partCpt    ),
      moistrhst_  ( moistrhst  ),
      oxidationt_ ( oxidaitont ),
      co2gasift_  ( co2gasift  ),
      h2ogasift_  ( h2ogasift  ),
      co2coratiot_( co2coratiot),
      inttempGt_  ( inttempGt  ),
      tempPt_     ( tempPt     )
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const{
      return new ParticleTemperatureRHS<FieldT>( partmast_, partCpt_, moistrhst_, oxidationt_, co2gasift_,
                                                 h2ogasift_, co2coratiot_, tempPt_, inttempGt_ );
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
ParticleTemperatureRHS( const Expr::Tag& partmast,
                        const Expr::Tag& partCpt,
                        const Expr::Tag& moistrhst,
                        const Expr::Tag& oxidationt,
                        const Expr::Tag& co2gasift,
                        const Expr::Tag& h2ogasift,
                        const Expr::Tag& co2coratiot,
                        const Expr::Tag& tempPt,
                        const Expr::Tag& inttempGt )
  : Expr::Expression<FieldT>(),
    alpha_( Coal::absored_heat_fraction_particle() )
{
  this->set_gpu_runnable(true);

  partmas_    = this->template create_field_request<FieldT>( partmast    );
  partCp_     = this->template create_field_request<FieldT>( partCpt     );
  moistrhs_   = this->template create_field_request<FieldT>( moistrhst   );
  oxidation_  = this->template create_field_request<FieldT>( oxidationt  );
  co2gasif_   = this->template create_field_request<FieldT>( co2gasift   );
  h2ogasif_   = this->template create_field_request<FieldT>( h2ogasift   );
  co2coratio_ = this->template create_field_request<FieldT>( co2coratiot );
  tempP_      = this->template create_field_request<FieldT>( tempPt      );
  inttempG_   = this->template create_field_request<FieldT>( inttempGt   );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleTemperatureRHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& partmas    = partmas_   ->field_ref();
  const FieldT& partCp     = partCp_    ->field_ref();
  const FieldT& moistrhs   = moistrhs_  ->field_ref();
  const FieldT& oxidation  = oxidation_ ->field_ref();
  const FieldT& co2gasif   = co2gasif_  ->field_ref();
  const FieldT& h2ogasif   = h2ogasif_  ->field_ref();
  const FieldT& co2coratio = co2coratio_->field_ref();
  const FieldT& tempP      = tempP_     ->field_ref();
  const FieldT& inttempG   = inttempG_  ->field_ref();

  const double hco  = 9629.64E3;  // J/kg
  const double hco2 = 33075.72E3; // J/kg [1]

  // temperature rhs for vaporization
  // @ T = 300 K ,Heat of Vaporization = 2438 kj/kg, Tcritical = 647.096 K
  const double Tc = 647.096;

  // temperature rhs for char oxidation
  result <<=
  (
      cond( tempP < Tc, 2438.E3 * pow( (1.0- tempP/Tc)/(1.0-300/Tc), 0.375 ) * moistrhs )( 0.0 )  // vaporization
      + -1.0 * alpha_ * ( oxidation * co2coratio / ( 1.0 + co2coratio) ) * hco2 // CO2
      + -1.0 * alpha_ * ( oxidation/ (1.0+ co2coratio) ) * hco // CO
      + alpha_ * ( co2gasif *14.37E6)                            // CO2 gasification reaction [2]
      + alpha_ * ( h2ogasif * 10.94E6)                           // H2O gasification reaction [2]
  )
  / (partmas * partCp);
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
