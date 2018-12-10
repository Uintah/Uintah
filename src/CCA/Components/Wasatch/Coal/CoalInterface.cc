#include <stdexcept>
#include <sstream>

#include <spatialops/particles/ParticleFieldTypes.h>

#include <expression/Functions.h>

#include <pokitt/thermo/Enthalpy.h>

#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>

#include <CCA/Components/Wasatch/Coal/VaporizationBoiling/EvapInterface.h>
#include <CCA/Components/Wasatch/Coal/CoalInterface.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharInterface.h>
#include <CCA/Components/Wasatch/Coal/ParticleTemperatureRHS.h>
#include <CCA/Components/Wasatch/Coal/CoalHeatCapacity.h>
#include <CCA/Components/Wasatch/Coal/ProducedGasComposition.h>
#include <CCA/Components/Wasatch/Coal/ConsumedGasComposition.h>
#include <CCA/Components/Wasatch/Coal/HeatFromCharReactions.h>

using std::vector;
using std::string;
using std::endl;
using std::ostringstream;
using std::set;
using std::map;

namespace Coal{

  typedef vector<string> StringVec;

  //--------------------------------------------------------------------

  DEV::DEVSpecies gas_coal2dev( const GasSpeciesName cspec )
  {
    DEV::DEVSpecies s;
    switch( cspec ){
    case CO2:  s=DEV::CO2;             break;
    case H2O:  s=DEV::H2O;             break;
    case CO :  s=DEV::CO;              break;
    case HCN:  s=DEV::HCN;             break;
    case NH3:  s=DEV::NH3;             break;
    case CH4:  s=DEV::CH4;             break;
    case H2:   s=DEV::H2;              break; 
    default:   s=DEV::INVALID_SPECIES; break;
    }
    return s;
  }

  //------------------------------------------------------------------

  CHAR::CharGasSpecies gas_coal2char( const GasSpeciesName cspec )
  {
    CHAR::CharGasSpecies s;
    switch( cspec ){
    case O2 : s=CHAR::O2;              break;
    case CO2: s=CHAR::CO2;             break;
    case CO : s=CHAR::CO;              break;
    case H2O: s=CHAR::H2O;             break;
    case H2 : s=CHAR::H2;              break;
    case CH4: s=CHAR::CH4;             break;
    default:  s=CHAR::INVALID_SPECIES; break;
    }
    return s;
  }

  //------------------------------------------------------------------

  EVAP::EvapSpecies gas_coal2vap( const GasSpeciesName cspec )
  {
    EVAP::EvapSpecies s;
    switch( cspec ){
    case H2O: s=EVAP::H2O;              break;
    default:  s=EVAP::INVALID_SPECIES;  break;
    }
    return s;
  }

    //------------------------------------------------------------------

  size_t
  get_species_index( const string name,
                     const StringVec& speciesNames )
  {
    for( size_t i=0; i<speciesNames.size(); ++i ){
      if( name == speciesNames[i] )
        return i;
    }
    std::ostringstream msg;
    msg << endl
        << __FILE__ << " : " << __LINE__ << endl
        << "species '" << name << "' was not found in list:" << endl;
    for( StringVec::const_iterator istr=speciesNames.begin(); istr!=speciesNames.end(); ++istr ){
      msg << "   " << *istr << endl;
    }
    msg << endl;
    throw std::invalid_argument( msg.str() );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  CoalInterface<FieldT>::
  CoalInterface( GraphCategories& gc,
                 const CoalType coalType,
                 const DEV::DevModel devModel,
                 const CHAR::CharModel chModel,
                 const Expr::Tag& pDiamTag,
                 const Expr::Tag& pTempTag,
                 const Expr::Tag& gTempTag,
                 const Expr::Tag& mixMWTag,
                 const Expr::Tag& pDensTag,
                 const Expr::Tag& gPressTag,
                 const Expr::Tag& pMassTag,
                 const Expr::Tag& rePtag,
                 const Expr::Tag& scGTag,
                 const SpeciesTagMap& specTagMap,
                 const Expr::Tag& pMass0Tag,
                 const Expr::Tag& pDens0Tag,
                 const Expr::Tag& pDiam0Tag )
    : gc_         ( gc ),
      pTempTag_   ( pTempTag  ),
      gTempTag_   ( gTempTag  ),
      mixMWTag_   ( mixMWTag  ),
      pDensTag_   ( pDensTag  ),
      gPressTag_  ( gPressTag ),
      pMassTag_   ( pMassTag  ),
      pDiamTag_   ( pDiamTag  ),
      rePTag_     ( rePtag    ),
      scGTag_     ( scGTag    ),
      pCpTag_     ( WasatchCore::TagNames::self().pHeatCapacity ),
      pTemp_rhsTag_( Coal::StringNames::self().coal_temprhs, Expr::STATE_NONE ),
      heatReleasedToGasTag_( Coal::StringNames::self().char_heattogas, Expr::STATE_NONE ),

      specTagMap_( specTagMap ),
      devModel_  ( devModel   ),

      dev_( new DEV::DevolatilizationInterface<FieldT>( gc, coalType, devModel, pTempTag, pMassTag, pMass0Tag) ),

      char_( new CHAR::CharInterface<FieldT>( gc, pDiamTag, pTempTag, gTempTag,
                                              get_species_tag(CO2),get_species_tag(CO),
                                              get_species_tag(O2), get_species_tag(H2),
                                              get_species_tag(H2O),get_species_tag(CH4),
                                              mixMWTag, pDensTag, gPressTag, pMassTag,
                                              pMass0Tag, pDens0Tag, pDiam0Tag,
                                              dev_->volatiles_tag(), coalType, devModel, chModel) ),

      evap_( new EVAP::EvapInterface<FieldT>( gc, gTempTag, pTempTag, pDiamTag, rePtag,
                                              scGTag, get_species_tag(H2O), mixMWTag,
                                              gPressTag, pMassTag, coalType) ),

      mvTag_      ( dev_ ->volatiles_tag()              ),
      charTag_    ( char_->char_mass_tag()              ),
      moistureTag_( evap_->retrieve_moisture_mass_tag() ),
      gas_        (CanteraObjects::get_gasmix()         )
  {
    // assemble the collection of tags corresponding to gas phase
    // source terms from the CPD and Char oxidation models,
    // respectively.
    const Expr::TagList cpdSpeciesSrcTags  = dev_->gas_species_src_tags();
    const Expr::Tag tarProdRTag_           = dev_->tar_production_rate_tag();
    
    for( Expr::TagList::const_iterator iss=cpdSpeciesSrcTags.begin(); iss!=cpdSpeciesSrcTags.end(); ++iss ){
      gasSpeciesSourceTags_.push_back( *iss );
    }

    // CharSpecRHSt[0]  : CO2 rhs consumption rate in gas phase
    // CharSpecRHSt[1]  : CO  rhs consumption rate in gas phase
    // CharSpecRHSt[2]  : O2  rhs consumption rate in gas phase
    // CharSpecRHSt[3]  : H2  rhs consumption rate in gas phase
    // CharSpecRHSt[4]  : H2O rhs consumption rate in gas phase
    const Expr::TagList charSpecRHSt = char_->gas_species_src_tags();
    for( Expr::TagList::const_iterator iss=charSpecRHSt.begin(); iss!=charSpecRHSt.end(); ++iss ){
      gasSpeciesSourceTags_.push_back( *iss );
    }

    // Basically in Evaporation only water vaporizes, So amount of water vaporized from particle goes to gas phase
    // Notice : for gas species source term it must be multiplied by -1 !
    const Expr::Tag evaporation_rhsTag = evap_->moisture_rhs_tag();
    gasSpeciesSourceTags_.push_back( evaporation_rhsTag );

    productionRateTags_.clear();
    productionRateTags_.push_back( dev_->volatile_consumption_rate_tag());
    productionRateTags_.push_back( char_->char_consumption_rate_tag()   );
    productionRateTags_.push_back( evap_->moisture_rhs_tag()            );

    register_expressions();
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  CoalInterface<FieldT>::
  ~CoalInterface()
  {
    delete dev_;
    delete char_;
    delete evap_;

    CanteraObjects::restore_gasmix(gas_);
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  CoalInterface<FieldT>::
  register_expressions()
  {
    Expr::ExpressionFactory& factory  = *(gc_[WasatchCore::ADVANCE_SOLUTION]->exprFactory);
    Expr::ExpressionFactory& iFactory = *(gc_[WasatchCore::INITIALIZATION  ]->exprFactory);
    proc0cout << "\nRegistering coal expressions for energy source terms" << std::endl;

    const Expr::TagList charGasTags   = char_->gas_species_src_tags(); // consist of three tags, 1.CO2_RHS (Negative), 2.CO_RHS (Negative), 3.O2_RHS (Positive)
    const Expr::TagList devSpecRHSTags = dev_->gas_species_src_tags();  // consist of 8 tags. ALl are Negative !
    const Expr::Tag     evapRHSTag    = evap_->moisture_rhs_tag();     // Evaporation RHS. Always negative !
    const Expr::Tag     oxidationTag  = char_->oxidation_tag();
    const Expr::Tag     co2GasifTag   = char_->char_gasification_co2_rate_tag();
    const Expr::Tag     h2oGasifTag   = char_->char_gasification_h2o_rate_tag();
    const Expr::Tag     co2CoRatioTag = char_->co2coratio_tag();
    const Expr::Tag     o2RHSTag      = char_->gas_species_src_tag(CHAR::O2);

    const StringNames& sNames = StringNames::self();

    typedef typename Expr::LinearFunction  <FieldT>::Builder LinFunBuilder;
    typedef typename Expr::ConstantExpr    <FieldT>::Builder ConstBuilder;
    typedef typename ProducedGasComposition<FieldT>::Builder ProducedCompBuilder;
    typedef typename ConsumedGasComposition<FieldT>::Builder ConsumedCompBuilder;
    typedef typename HeatFromCharReactions <FieldT>::Builder deltaHCharBuilder;

    typedef typename pokitt::Enthalpy     <FieldT>::Builder EnthalpyBuilder;
    typedef typename ExprAlgebra          <FieldT>::Builder AlgebraBuilder;
    typedef typename CoalHeatCapacity     <FieldT>::Builder CoalCpBuilder;
    typedef          ExprAlgebra          <FieldT>          Algebra;

    /* register expressions corresponding to gas *consumed* at the particle */
    /*****************************************************************************************/
    const Expr::Tag zeroTag                  ( "particle-zero-val"               , Expr::STATE_NONE );
    const Expr::Tag totalConsumptionRateTag  ( sNames.coalTotalGasConsumptionRate, Expr::STATE_NONE );
    const Expr::Tag consumedGasEnthalpyTag   ( sNames.coalConsumedGasEnthalpy    , Expr::STATE_NONE );
    const Expr::Tag consumedGasEnthalpySrcTag( sNames.coalConsumedGasEnthalpySrc , Expr::STATE_NONE );

    factory.register_expression( new ConstBuilder( zeroTag, 0.) );

    /* Gasification rates are given in (mass char produced)/second and need to be converted to
     * (species mass)/second consumed
     */
    // Expression for O2 consumption rate registered in the implemented char model interface
    const std::string&  consumedYiPrefix = sNames.coalConsumedGasYi;
    const SpeciesTagMap consumedYiTags   = ConsumedCompBuilder::
                                           get_consumed_species_tag_map( consumedYiPrefix );

    /* Assemble a TagList of all species in the cantera species group. If a species is not
     * present in consumedYiTags, a tag to a zero-valued field will be added in its place.
     */
    Expr::TagList allYiTagsConsumed;
    for(const std::string& specName : gas_->speciesNames()){
      const GasSpeciesName                specEnum = gas_name_to_enum   ( specName );
      const SpeciesTagMap::const_iterator iter     = consumedYiTags.find( specEnum );

      allYiTagsConsumed.push_back( iter == consumedYiTags.end() ? zeroTag : iter->second );
    }

    factory.register_expression( new ConsumedCompBuilder( consumedYiPrefix,
                                                          totalConsumptionRateTag,
                                                          o2RHSTag,
                                                          co2GasifTag,
                                                          h2oGasifTag ));

    factory.register_expression( new EnthalpyBuilder( consumedGasEnthalpyTag, gTempTag_,
                                                      allYiTagsConsumed) );

    /* this calculates the enthalpy removed from the gas (J/s) from gas consumption
     * at the particle.
     */
    factory.register_expression( new AlgebraBuilder( consumedGasEnthalpySrcTag,
                                                     Expr::tag_list( totalConsumptionRateTag,
                                                                     consumedGasEnthalpyTag ),
                                                     Algebra::PRODUCT ) );

    /* register expressions corresponding to gas *produced* at the particle */
    /*****************************************************************************************/

    const Expr::Tag totalProductionRateTag   (sNames.coalTotalGasProductionRate, Expr::STATE_NONE);
    const Expr::Tag producedGasEnthalpyTag   (sNames.coalProducedGasEnthalpy   , Expr::STATE_NONE);
    const Expr::Tag producedGasEnthalpySrcTag(sNames.coalProducedGasEnthalpySrc, Expr::STATE_NONE);

    const std::string& producedYiPrefix = sNames.coalProducedGasYi;
    const SpeciesTagMap producedYiTags = ProducedCompBuilder::
                                         get_produced_species_tag_map( producedYiPrefix,
                                                                       devModel_ );

    /* Assemble a TagList of all species in the cantera species group. If a species is not
     * present in producedYiTags, a tag to a zero-valued field will be added in its place.
     */
    Expr::TagList allYiTagsProduced;
    for(const std::string& specName : gas_->speciesNames()){
      const GasSpeciesName                specEnum = gas_name_to_enum   ( specName );
      const SpeciesTagMap::const_iterator iter     = producedYiTags.find( specEnum );

      allYiTagsProduced.push_back( iter == producedYiTags.end() ? zeroTag : iter->second );
    }

        factory.register_expression( new ProducedCompBuilder( producedYiPrefix, totalProductionRateTag,
                                                              devSpecRHSTags, evapRHSTag,
                                                              oxidationTag, co2CoRatioTag,
                                                              co2GasifTag, h2oGasifTag,
                                                              devModel_ ) );

        factory.register_expression( new EnthalpyBuilder( producedGasEnthalpyTag, pTempTag_,
                                                          allYiTagsProduced ) );

        /* this calculates the enthalpy removed from the gas (J/s) from gas production
         * at the particle.
         */
        factory.register_expression( new AlgebraBuilder( producedGasEnthalpySrcTag,
                                                         Expr::tag_list( totalProductionRateTag,
                                                                         producedGasEnthalpyTag ),
                                                         Algebra::PRODUCT ) );


    /*****************************************************************************************/
    const Expr::Tag heatFromCharRxnTag     ( sNames.heatFromCharRxns,             Expr::STATE_NONE );
    const Expr::Tag heatFromCharRxnToGasTag( sNames.heatFromCharRxns + "_to_gas", Expr::STATE_NONE );

    factory.register_expression( new deltaHCharBuilder( heatFromCharRxnTag,
                                                        oxidationTag,
                                                        co2GasifTag,
                                                        h2oGasifTag,
                                                        co2CoRatioTag ) );

    factory.register_expression( new LinFunBuilder( heatFromCharRxnToGasTag,
                                                    heatFromCharRxnTag,
                                                    1. - Coal::absored_heat_fraction_particle(),
                                                    0 ) );

    factory.register_expression( new AlgebraBuilder( heatReleasedToGasTag_,
                                                     Expr::tag_list( consumedGasEnthalpySrcTag,
                                                                     producedGasEnthalpySrcTag,
                                                                     heatFromCharRxnToGasTag ),
                                                     Algebra::SUM ) );

    Expr::TagList charco2co;
    charco2co.push_back(char_->gas_species_src_tag(CHAR::CO2));
    charco2co.push_back(char_->gas_species_src_tag(CHAR::CO ));

    // AllowOverwrite = true for CoalHeatCapacity because a placeholder expression is registered first
    // (in PArticleTemperatureEquation.cc).
    factory. register_expression( new CoalCpBuilder( pCpTag_,
                                                     mvTag_,
                                                     charTag_,
                                                     moistureTag_,
                                                     pMassTag_,
                                                     pTempTag_ ) );

    iFactory.register_expression( new CoalCpBuilder( pCpTag_,
                                                     mvTag_,
                                                     charTag_,
                                                     moistureTag_,
                                                     pMassTag_,
                                                     pTempTag_ ) );

    factory.register_expression( new typename ParticleTemperatureRHS <FieldT>::
                                 Builder( pTemp_rhsTag_,
                                          pMassTag_,
                                          pCpTag_,
                                          evapRHSTag,
                                          pTempTag_,
                                          heatFromCharRxnTag ) );

    // Plug the char production rate from CPD into the char model.
    // The CPD model produces char.  The char model calculates the
    // char consumption rate as a positive quantity. Therefore, we
    // subtract the CPD contribution from the char consumption rate.
    factory.attach_dependency_to_expression( dev_->char_production_rate_tag(),
                                             char_->char_consumption_rate_tag(),
                                             Expr::SUBTRACT_SOURCE_EXPRESSION );
    // any other couplings between sub-models should be done here.
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Coal::CoalEqVec
  CoalInterface<FieldT>::
  get_equations() const
  {
    Coal::CoalEqVec evapEqns = evap_->get_equations();
    Coal::CoalEqVec devEqns  = dev_ ->get_equations();
    Coal::CoalEqVec charEqns = char_->get_equations();

    Coal::CoalEqVec eqns; eqns.clear();

    eqns.insert( eqns.end(), evapEqns.begin(), evapEqns.end() );
    eqns.insert( eqns.end(), devEqns. begin(), devEqns. end() );
    eqns.insert( eqns.end(), charEqns.begin(), charEqns.end() );

    return eqns;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  const Expr::Tag&
  CoalInterface<FieldT>::
  get_species_tag( const GasSpeciesName spec ) const
  {
    const SpeciesTagMap::const_iterator i=specTagMap_.find(spec);
    if( i==specTagMap_.end() ){
      ostringstream msg;
      msg << endl
          << "Error from get_species_tag()" << endl
          << __FILE__ << " : " << __LINE__ << endl
          << "No species tag found for '" << species_name(spec) << "'" << endl
          << endl;
      throw std::runtime_error( msg.str() );
    }
    return i->second;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::TagList
  CoalInterface<FieldT>::
  gas_species_source_term( const GasSpeciesName spec,
                           const bool forceMatch ) const
  {
    Expr::TagList tags;

    const Expr::Tag devTag  = dev_ ->gas_species_src_tag( gas_coal2dev(spec) );
    const Expr::Tag charTag = char_->gas_species_src_tag( gas_coal2char(spec));
    const Expr::Tag evapTag = evap_->gas_species_src_tag( gas_coal2vap(spec) );

    if( devTag  != Expr::Tag() ) tags.push_back( devTag  );
    if( charTag != Expr::Tag() ) tags.push_back( charTag );
    if( evapTag != Expr::Tag() ) tags.push_back( evapTag );

    if( tags.empty() && forceMatch ){
      ostringstream msg;
      msg << endl
          << "In gas_species_source_term()" << endl
          << __FILE__ << " : " << __LINE__ << endl
          << "No source term was found for species '" << species_name(spec) << "'" << endl
          << endl;
      throw std::runtime_error( msg.str() );
    }
    return tags;
  }
  
  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::Tag
  CoalInterface<FieldT>::
  tar_source_term() const
  {
    return dev_->tar_production_rate_tag();
  }
  

  //==================================================================
  // Explicit template instantiation
  template class CoalInterface< SpatialOps::Particle::ParticleField >;
  //==================================================================

} // namespace coal
