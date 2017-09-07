#ifndef HeatReleasedtoGas_CHAR_h
#define HeatReleasedtoGas_CHAR_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/KobayashiSarofim/KobSarofimData.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/CPDData.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

#include <cantera/IdealGasMix.h>
#include <pokitt/CanteraObjects.h> // include cantera wrapper


/**
 *   Author : Babak Goshayeshi (www.bgoshayeshi.com)
 *   Date   : March 2 2011
 *   University of Utah - Institute for Clean and Secure Energy
 *
 *   Heat that released to the gas (j/s) which depend on :
 *
 *   co2corhst  : co2 and co consumption rate (kg/s)
 *   o2rhst     : O2 consumption rate in gas ( kg/s)
 *   devspecrhst: CPD productions species ( consumption for gas phase ) kg/s
 *   evaprhst   : moisture evaporation rate (consumption in gas ) kg/s
 *   oxidationt : char oxidation rhs tag
 *   co2coratiot: co2/co ratio tag
 *   co2gasift  : char consumption due to co2 gasification reaction
 *   h2ogasift  : char consumption due to h2o gasification reaction
 *   tempPt     : Particle Temperature k
 *   tempGt     : Gas Temperature k
 *   gaspresst  : Gas pressure (pa)
 *
 */
namespace Coal {

  void canteraSpecCheck( const bool haveRegSpec,
                         const bool haveCpdSpec,
                         const DEV::DevModel devModel )
  {
    // Check if an error is present.
    bool haveCpdError = ( devModel == DEV::CPDM && haveCpdSpec == false );
    bool haveError    = ( haveCpdError || haveRegSpec == false );

    if( haveError ){

      std::ostringstream msg;

      msg   << __FILE__ << " : " << __LINE__ << std::endl
          << "\nOne or more of the species used to calculate mass &  " << std::endl
          << "energy exchange terms (particle <--> gas)  is missing  " << std::endl
          << "from the Cantera species group. Please ensure that the " << std::endl
          << "Cantera input file includes all of the following       " << std::endl
          << "species:                                               " << std::endl
          << std::endl
          << "CO2  CO  H2O  O2  H2  ";
      if( haveCpdError ){
        msg << "H  HCN  NH3  CH4  "
            << std::endl
            << std::endl;
      }
      else{
        msg << std::endl
            << std::endl;
      }

      throw std::runtime_error( msg.str() );
    }
  }

  template< typename FieldT >
class HeatReleasedtoGas
 : public Expr::Expression<FieldT>
{
  typedef typename std::vector<typename FieldT::const_iterator> VecIterator;

  DECLARE_FIELDS( FieldT, o2rhs_, evaprhs_, tempP_, tempG_, gaspress_ )
  DECLARE_FIELDS( FieldT, oxidation_, co2coratio_, co2gasif_, h2ogasif_ )

  DECLARE_VECTOR_OF_FIELDS( FieldT, cpdspecrhs_ )

  bool haveRegSpecies_, haveCpdSpecies_;
  int iCO2_, iCO_, iH2O_, iO2_, iH2_, iHCN_, iNH3_, iCH4_, iH_;
  const DEV::DevModel dvmodel_;
  const double alpha_;

  HeatReleasedtoGas( const Expr::Tag&     o2rhst,
                     const Expr::TagList& devspecrhst,
                     const Expr::Tag&     evaprhst,
                     const Expr::Tag&     oxidationt,
                     const Expr::Tag&     co2coratiot,
                     const Expr::Tag&     co2gasift,
                     const Expr::Tag&     h2ogasift,
                     const Expr::Tag&     tempPt,
                     const Expr::Tag&     tempGt,
                     const Expr::Tag&     gaspresst,
                     const DEV::DevModel dvmodel);
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param   co2corhst   co2 and co consumption rate (kg/s)
     *  \param   o2rhst      O2 consumption rate in gas ( kg/s)
     *  \param   devspecrhst CPD productions species ( consumption for gas phase ) kg/s
     *  \param   evaprhst    moisture evaporation rate (consumption in gas ) kg/s
     *  \param   tempPt      Partile Temperature k
     *  \param   tempGt      Gas Temperature k
     *  \param   gaspresst   Gas pressure (pa)
     */
    Builder( const Expr::Tag&     heatRelTag,
             const Expr::Tag&     o2rhst,
             const Expr::TagList& devspecrhst,
             const Expr::Tag&     evaprhst,
             const Expr::Tag&     oxidationt,
             const Expr::Tag&     co2coratiot,
             const Expr::Tag&     co2gasift,
             const Expr::Tag&     h2ogasift,
             const Expr::Tag&     tempPt,
             const Expr::Tag&     tempGt,
             const Expr::Tag&     gaspresst,
             const DEV::DevModel dvmodel);

    Expr::ExpressionBase* build() const;

  private:
    const Expr::TagList devspecrhst_;
    const Expr::Tag o2rhst_, evaprhst_, oxidationt_, co2coratiot_, co2gasift_,
                    h2ogasift_, tempPt_, tempGt_, gaspresst_;
    const DEV::DevModel dvmodel_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



  template< typename FieldT >
  HeatReleasedtoGas<FieldT>::
  HeatReleasedtoGas(const Expr::Tag&     o2rhst,
                    const Expr::TagList& devspecrhst,
                    const Expr::Tag&     evaprhst,
                    const Expr::Tag&     oxidationt,
                    const Expr::Tag&     co2coratiot,
                    const Expr::Tag&     co2gasift,
                    const Expr::Tag&     h2ogasift,
                    const Expr::Tag&     tempPt,
                    const Expr::Tag&     tempGt,
                    const Expr::Tag&     gaspresst,
                    const DEV::DevModel dvmodel)
    : Expr::Expression<FieldT>(),
      dvmodel_    ( dvmodel     ),
      alpha_      ( 1.0 - Coal::absored_heat_fraction_particle())
  {
    Cantera::IdealGasMix* const gas = CanteraObjects::get_gasmix();


    // Set species indeces and check if each species is in the Cantera Group.
    haveRegSpecies_ = true;
    iCO2_ = gas->speciesIndex("CO2");    haveRegSpecies_ = ( haveRegSpecies_ && (iCO2_ != -1) );
    iCO_  = gas->speciesIndex("CO" );    haveRegSpecies_ = ( haveRegSpecies_ && (iCO_  != -1) );
    iH2O_ = gas->speciesIndex("H2O");    haveRegSpecies_ = ( haveRegSpecies_ && (iH2O_ != -1) );
    iO2_  = gas->speciesIndex("O2" );    haveRegSpecies_ = ( haveRegSpecies_ && (iO2_  != -1) );
    iH2_  = gas->speciesIndex("H2" );    haveRegSpecies_ = ( haveRegSpecies_ && (iH2_  != -1) );

    haveCpdSpecies_ = true;
    iHCN_  = gas->speciesIndex("HCN" );  haveCpdSpecies_ = ( haveCpdSpecies_ &&  (iHCN_ != -1) );
    iNH3_  = gas->speciesIndex("NH3" );  haveCpdSpecies_ = ( haveCpdSpecies_ &&  (iNH3_ != -1) );
    iCH4_  = gas->speciesIndex("CH4" );  haveCpdSpecies_ = ( haveCpdSpecies_ &&  (iCH4_ != -1) );
    iH_    = gas->speciesIndex("H"   );  haveCpdSpecies_ = ( haveCpdSpecies_ &&  (iH_   != -1) );

    canteraSpecCheck( haveRegSpecies_, haveCpdSpecies_, dvmodel );

    CanteraObjects::restore_gasmix(gas);

    evaprhs_   = this->template create_field_request<FieldT>( evaprhst    );
    oxidation_ = this->template create_field_request<FieldT>( oxidationt  );
    co2coratio_= this->template create_field_request<FieldT>( co2coratiot );
    co2gasif_  = this->template create_field_request<FieldT>( co2gasift   );
    h2ogasif_  = this->template create_field_request<FieldT>( h2ogasift   );
    tempP_     = this->template create_field_request<FieldT>( tempPt      );
    tempG_     = this->template create_field_request<FieldT>( tempGt      );
    gaspress_  = this->template create_field_request<FieldT>( gaspresst   );
    o2rhs_     = this->template create_field_request<FieldT>( o2rhst      );

    this->template create_field_vector_request<FieldT>( devspecrhst, cpdspecrhs_ );
  }

//--------------------------------------------------------------------

template< typename FieldT >
void
HeatReleasedtoGas<FieldT>::
evaluate()
{
  const double mwco  = 28.0;
  const double mwco2 = 44.0;
  const double mwh2o = 18.0;
  const double mwchar= 12.0;
  const double mwh2  = 2.0;

  using namespace SpatialOps;
  FieldT& result = this->value();

  // Char oxidation part
  double totalmass;
  double oxid_co2, oxid_co;
  const double hco  = 9629.64E3;  // J/kg
  const double hco2 = 33075.72E3; // J/kg [1]

  // Enthalpy carried by mass flux
  Cantera::IdealGasMix* const gas = CanteraObjects::get_gasmix();
  const int kk = gas->nSpecies();
  std::vector<double> Wmass(kk,0.0);

  typename FieldT::const_iterator itempP     = tempP_     ->field_ref().begin();
  typename FieldT::const_iterator itempG     = tempG_     ->field_ref().begin();
  typename FieldT::const_iterator ipress     = gaspress_  ->field_ref().begin();
  typename FieldT::const_iterator icharo2    = o2rhs_     ->field_ref().begin();
  typename FieldT::const_iterator ievap      = evaprhs_   ->field_ref().begin();
  typename FieldT::const_iterator ioxidation = oxidation_ ->field_ref().begin();
  typename FieldT::const_iterator ico2gasif  = co2gasif_  ->field_ref().begin();
  typename FieldT::const_iterator ih2ogasif  = h2ogasif_  ->field_ref().begin();
  typename FieldT::const_iterator ico2cor    = co2coratio_->field_ref().begin();
  typename FieldT::iterator       iresult    = result.begin();
  typename FieldT::iterator       iresult2   = result.begin();

  VecIterator Veci;  // used for CPD

  for( size_t i=0; i<cpdspecrhs_.size(); ++i ){
    Veci.push_back( cpdspecrhs_[i]->field_ref().begin() );
  }

  totalmass = 0;
  for( ; iresult != result.end(); ++itempP, ++itempG, ++ipress,
       ++icharo2, ++ievap, ++ioxidation, ++ico2gasif, ++ih2ogasif,
       ++ico2cor ,++iresult ){

    //_________________________________________________
    // compute contributions from the surface reactions
    // (gasification, char oxidation)

    oxid_co2 = *ioxidation * (*ico2cor/(1.0+ *ico2cor));
    oxid_co  = *ioxidation / (1.0+ *ico2cor) ;

    // Char Oxidation - Exothermic
    *iresult = -1 * alpha_ * ( oxid_co2 * hco2 + oxid_co * hco );

    // CO2 Gasification reaction - Endothermic
    *iresult += alpha_ * ( *ico2gasif * 14.37E6 );

    // H2O Gasification reaction - Endothermic
    *iresult += alpha_ * ( *ih2ogasif * 10.94E6 );

    //_____________________________________________
    // compute contribution from mass coming to the
    // particle from the gas phase (participating
    // in gasification & char oxidation)
    // O2, H2O and CO2 come to the particle at gas temperature

   totalmass = *icharo2 + (*ico2gasif / mwchar * -mwco2) + (*ih2ogasif/mwchar* -mwh2o);

    if (totalmass > 0.0) {
      for( std::vector<double>::iterator i=Wmass.begin(); i!=Wmass.end(); ++i ) *i=0;
      Wmass[iO2_ ] = *icharo2/totalmass;
      Wmass[iCO2_] = (*ico2gasif/ mwchar * -mwco2)/totalmass;
      Wmass[iH2O_] = (*ih2ogasif/mwchar* -mwh2o)/totalmass;
      gas->setState_TPY(*itempG,*ipress,&Wmass[0]);
      *iresult -= totalmass * (gas->enthalpy_mass());
    }

    //_____________________________________________________
    // compute contributions from mass carrying energy from
    // the particle due to char oxidation and gasification

    // CO2 and CO form CharOxidation
    // H2O from Evaporation and CPD
    // CO and H2 from gasification reaction
    // even with using setMassFractions_NoNorm still totalmass is required for ethalpy.

    oxid_co  *= mwco/mwchar;
    oxid_co2 *= mwco2/mwchar;

    totalmass = (oxid_co + oxid_co2 + *ico2gasif/mwchar*mwco*2.0 + *ih2ogasif/mwchar*mwco
                + *ih2ogasif/mwchar*mwh2 + *ievap);

    typename VecIterator::iterator iveci = Veci.begin();
    typename FieldT::const_iterator ispecF = *iveci;
    for( int i=0; iveci != Veci.end(); ++iveci, ++i ){
      totalmass += **iveci;
    }
    if( totalmass ==0.0 ){
      iresult2=iresult;
      ++iresult2;
      if (iresult2 != result.end()) {
        for (iveci = Veci.begin(); iveci != Veci.end(); ++iveci) {
          ispecF = *iveci;
          ++ispecF;
          *iveci = ispecF;
        }
      }
      continue;
    }

    //______________________________________________________________
    // compute contributions to mass-related energy flux due to mass
    // coming from the CPD model (leaves at the particle temperature)
    for( std::vector<double>::iterator i=Wmass.begin(); i!=Wmass.end(); ++i ) *i=0;

    //Devolatilization
    switch (dvmodel_) {
      case DEV::CPDM:
        Wmass[iCO2_ ] = (oxid_co2 + *(Veci[CPD::CO2])) / totalmass;
        Wmass[iCO_  ] = (oxid_co + *ico2gasif/mwchar*mwco*2.0 + *ih2ogasif/mwchar*mwco
            + *(Veci[CPD::CO ])) / totalmass;
        Wmass[iH2O_ ] = (*ievap + *(Veci[CPD::H2O])) / totalmass;
        Wmass[iH2_  ] = (*ih2ogasif/mwchar*mwh2)     / totalmass;
        Wmass[iHCN_ ] = *(Veci[CPD::HCN])            / totalmass;
        Wmass[iNH3_ ] = *(Veci[CPD::NH3])            / totalmass;
        Wmass[iCH4_ ] = *(Veci[CPD::CH4])            / totalmass;
        Wmass[iH_   ] = *(Veci[CPD::H])              / totalmass;

        break;

      case DEV::KOBAYASHIM:
        Wmass[iCO2_ ] = oxid_co2 / totalmass;
        Wmass[iCO_  ] = (oxid_co + *ico2gasif/mwchar*mwco*2.0 + *ih2ogasif/mwchar*mwco
            + *(Veci[SAROFIM::CO ]))  / totalmass;
        Wmass[iH2O_ ] = *ievap                     / totalmass;
        Wmass[iH2_  ] = (*ih2ogasif/mwchar*mwh2
            + *(Veci[SAROFIM::H2]))    / totalmass;

        break;

      case DEV::DAE:
      case DEV::SINGLERATE:
        Wmass[iCO2_ ] = oxid_co2 / totalmass;
        Wmass[iCO_  ] = (oxid_co + *ico2gasif/mwchar*mwco*2.0 + *ih2ogasif/mwchar*mwco
            + *(Veci[SNGRATE::CO ]))  / totalmass;
        Wmass[iH2O_ ] = *ievap                     / totalmass;
        Wmass[iH2_  ] = (*ih2ogasif/mwchar*mwh2
            + *(Veci[SNGRATE::H2]))    / totalmass;

        break;

      case DEV::INVALID_DEVMODEL:
        throw std::invalid_argument( "Invalid devolatilization model in HeatReleasedtoGas::evaluate()" );
    }


    gas->setState_TPY(*itempP,*ipress,&Wmass[0]);

    *iresult -= totalmass * (gas->enthalpy_mass()); // totalmass that is negative (Gas Consumption)

    iresult2=iresult;
    ++iresult2;
    if (iresult2 != result.end()) {
      for (iveci = Veci.begin(); iveci != Veci.end(); ++iveci) {
        ispecF = *iveci;
        ++ispecF;
        *iveci = ispecF;
      }
    }

  }
  CanteraObjects::restore_gasmix(gas);
}

//--------------------------------------------------------------------

template< typename FieldT >
HeatReleasedtoGas<FieldT>::
Builder::Builder( const Expr::Tag&     heatRelTag,
                  const Expr::Tag&     o2rhst,
                  const Expr::TagList& devspecrhst,
                  const Expr::Tag&     evaprhst,
                  const Expr::Tag&     oxidationt,
                  const Expr::Tag&     co2coratiot,
                  const Expr::Tag&     co2gasift,
                  const Expr::Tag&     h2ogasift,
                  const Expr::Tag&     tempPt,
                  const Expr::Tag&     tempGt,
                  const Expr::Tag&     gaspresst,
                  const DEV::DevModel dvmodel )
  : ExpressionBuilder(heatRelTag),
    devspecrhst_( devspecrhst ),
    o2rhst_     ( o2rhst      ),
    evaprhst_   ( evaprhst    ),
    oxidationt_ ( oxidationt  ),
    co2coratiot_( co2coratiot ),
    co2gasift_  ( co2gasift   ),
    h2ogasift_  ( h2ogasift   ),
    tempPt_     ( tempPt      ),
    tempGt_     ( tempGt      ),
    gaspresst_  ( gaspresst   ),
    dvmodel_    ( dvmodel     )
{
}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
HeatReleasedtoGas<FieldT>::Builder::build() const
{
  return new HeatReleasedtoGas<FieldT>(o2rhst_, devspecrhst_, evaprhst_, oxidationt_, co2coratiot_,
                                       co2gasift_, h2ogasift_, tempPt_, tempGt_, gaspresst_, dvmodel_);
}

} // namespace CHAR
/*
 [1] The Combustion Rates of Coal Chars : A Review, L. W. Smith, symposium on combustion 19, 1982, pp 1045-1065
 */
#endif // HeatReleasedtoGas_CHAR_h

