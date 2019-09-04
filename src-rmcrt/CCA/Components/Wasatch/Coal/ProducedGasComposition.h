#ifndef ProducedGasComposition_h
#define ProducedGasComposition_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/KobayashiSarofim/KobSarofimData.h>
#include <CCA/Components/Wasatch/Coal/Devolatilization/CPD/CPDData.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

#include <cantera/IdealGasMix.h>
#include <pokitt/CanteraObjects.h> // include cantera wrapper


/**
 *   \class ProducedGasComposition
 *   \Author Josh McConnell
 *   \Date   March 2018
 *   \brief Calculates the species mass fractions and total production rate (kg/s)
 *          of gas produced at the particle. The rates inputed to this expression
 *          are "consumption" rates so all rate values should be negative.
 *
 *   \param devSpecRHSTags: species devolatilization rate tags
 *   \param evapRHSTag    : moisture evaporation rate tag
 *   \param charOxidTag   : char oxidation rhs tag
 *   \param co2coRatioTag : molar co2/co ratio tag
 *   \param co2GasifTag   : char consumption due to co2 gasification reaction
 *   \param h2oGasifTag   : char consumption due to h2o gasification reaction
 *
 */
namespace Coal {

//--------------------------------------------------------------------

std::vector<GasSpeciesName>
get_produced_species( const DEV::DevModel devModel ){
  std::vector<GasSpeciesName> species;

  //species that are always produced
  species.push_back( CO  ); // produced by oxidation & gasification
  species.push_back( CO2 ); // produced by oxidation
  species.push_back( H2O ); // produced by evaporation
  species.push_back( H2  ); // produced by H2O gasification

  switch (devModel){
    case DEV::CPDM:
      species.push_back( CH4 );
      species.push_back( NH3 );
      species.push_back( HCN );
      break;

    case DEV::KOBAYASHIM:
    case DEV::DAE:
    case DEV::SINGLERATE:
      // right now these models only produce CO, H2 and tar
      break;

    case DEV::INVALID_DEVMODEL:
      throw std::invalid_argument( "Invalid devolatilization model in ProducedGasComposition::evaluate()" );
  }

  return species;
}

//--------------------------------------------------------------------

  template< typename FieldT >
class ProducedGasComposition
 : public Expr::Expression<FieldT>
{

  DECLARE_FIELDS( FieldT, o2RHS_, evapRHS_ )
  DECLARE_FIELDS( FieldT, charOxidRHS_, co2coRatio_, co2GasifRHS_, h2oGasifRHS_ )

  DECLARE_VECTOR_OF_FIELDS( FieldT, devSpecRHS_ )

  const DEV::DevModel devModel_;

  const SpeciesTagMap& specTagMap_;

  const std::vector<GasSpeciesName> specEnums_;

  Cantera::IdealGasMix* const gas_;

  const double mwChar_, mwH2O_, mwCO2_, mwCO_, mwH2_;

  const size_t nSpec_;

  ProducedGasComposition( const Expr::TagList& devSpecRHS,
                          const Expr::Tag&     evapRHSTag,
                          const Expr::Tag&     charOxidRHSTag,
                          const Expr::Tag&     co2coRatioTag,
                          const Expr::Tag&     co2GasifRHSTag,
                          const Expr::Tag&     h2oGasifRHSTag,
                          const DEV::DevModel  devModel,
                          const SpeciesTagMap& specTagMap );
public:
  ~ProducedGasComposition();

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *   \param tagPrefix        : prefix for species mass fraction tags set here
     *   \param totalMassSrcTag  : tag for total mass production rate. The corresponding
     *                             field should always have non-positive values.
     *   \param devSpecRHSTag    : species devolatilization rate tags
     *   \param evapRHSTag       : moisture evaporation rate tag
     *   \param charOxidTag      : char oxidation rhs tag
     *   \param co2coRatioTag    : molar co2/co ratio tag
     *   \param co2GasifTag      : char consumption due to co2 gasification reaction
     *   \param h2oGasifTag      : char consumption due to h2o gasification reaction
     *
     */
    Builder( const std::string    tagPrefix,
             const Expr::Tag&     totalMassSrcTag,
             const Expr::TagList& devSpecRHSTag,
             const Expr::Tag&     evapRHSTag,
             const Expr::Tag&     charOxidRHSTag,
             const Expr::Tag&     co2coRatioTag,
             const Expr::Tag&     co2GasifRHSTag,
             const Expr::Tag&     h2oGasifRHSTag,
             const DEV::DevModel  devModel );

    Expr::ExpressionBase* build() const;

    static SpeciesTagMap get_produced_species_tag_map( const std::string tagPrefix,
                                                       DEV::DevModel     devmodel );

    static Expr::TagList get_produced_species_tags( const std::string tagPrefix,
                                                    DEV::DevModel      devmodel );

  private:
    Expr::TagList devSpecRHSTags_;
    const Expr::Tag evapRHSTag_, charOxidRHSTag_, co2coRatioTag_,
                    co2GasifRHSTag_, h2oGasifRHSTag_;
    const DEV::DevModel devModel_;
    const SpeciesTagMap specTagMap_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



  template< typename FieldT >
  ProducedGasComposition<FieldT>::
  ProducedGasComposition( const Expr::TagList& devSpecRHSTag,
                          const Expr::Tag&     evapRHSTag,
                          const Expr::Tag&     charOxidRHSTag,
                          const Expr::Tag&     co2coRatioTag,
                          const Expr::Tag&     co2GasifRHSTag,
                          const Expr::Tag&     h2oGasifRHSTag,
                          const DEV::DevModel  devModel,
                          const SpeciesTagMap& specTagMap )
    : Expr::Expression<FieldT>(),
      devModel_  ( devModel                                           ),
      specTagMap_( specTagMap                                         ),
      specEnums_ ( get_produced_species( devModel )                   ),
      gas_       ( CanteraObjects::get_gasmix()                       ),
      mwChar_    ( gas_->atomicWeight   ( gas_->elementIndex("C"  ) ) ),
      mwH2O_     ( gas_->molecularWeight( gas_->speciesIndex("H2O") ) ),
      mwCO2_     ( gas_->molecularWeight( gas_->speciesIndex("CO2") ) ),
      mwCO_      ( gas_->molecularWeight( gas_->speciesIndex("CO" ) ) ),
      mwH2_      ( gas_->molecularWeight( gas_->speciesIndex("H2" ) ) ),
      nSpec_     ( specTagMap_.size()                                 )
  {
    this->set_gpu_runnable(true);

    for( const GasSpeciesName& specEnum : specEnums_ ){
      std::string specName  = species_name( specEnum );
      int specIndex = gas_->speciesIndex( specName );

      if( specIndex == -1 ){
        std::ostringstream msg;

        msg   << __FILE__ << " : " << __LINE__ << std::endl
            << "One or more of the species used to calculate mass &    " << std::endl
            << "energy exchange terms (particle <--> gas)  is missing  " << std::endl
            << "from the Cantera species group. Please ensure that the " << std::endl
            << "Cantera input file includes all of the following       " << std::endl
            << "species:                                               " << std::endl
            << std::endl;

        for( const GasSpeciesName& specEnum2 : specEnums_ ){
          msg << species_name(specEnum2) << std::endl;
        }
        msg << std::endl;

        throw std::runtime_error( msg.str() );
      }
    }

    evapRHS_    = this->template create_field_request<FieldT>( evapRHSTag     );
    charOxidRHS_= this->template create_field_request<FieldT>( charOxidRHSTag );
    co2coRatio_ = this->template create_field_request<FieldT>( co2coRatioTag  );
    co2GasifRHS_= this->template create_field_request<FieldT>( co2GasifRHSTag );
    h2oGasifRHS_= this->template create_field_request<FieldT>( h2oGasifRHSTag );

    this->template create_field_vector_request<FieldT>( devSpecRHSTag, devSpecRHS_ );
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  ProducedGasComposition<FieldT>::
  ~ProducedGasComposition()
  {
    CanteraObjects::restore_gasmix(gas_);
  }

//--------------------------------------------------------------------

template< typename FieldT >
void
ProducedGasComposition<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typename Expr::Expression<FieldT>::ValVec& resultVec = this->get_value_vec();

  typename std::map<GasSpeciesName, SpatFldPtr<FieldT>> composition;
  for( SpeciesTagMap::const_iterator tIter = specTagMap_.cbegin();
      tIter != specTagMap_.cend(); ++tIter )
  {
    const GasSpeciesName specName = tIter->first;

    const size_t index = (size_t) std::distance(specTagMap_.cbegin(), tIter);

    composition[ specName ] = resultVec[index];

    *composition[ specName ] <<= 0.;
  }

  assert( composition.find(CO ) != composition.end() );
  assert( composition.find(CO2) != composition.end() );
  assert( composition.find(H2O) != composition.end() );
  assert( composition.find(H2 ) != composition.end() );

  FieldT& totalMass = *resultVec[nSpec_];

  const FieldT& evapRHS     = evapRHS_    ->field_ref();
  const FieldT& charOxidRHS = charOxidRHS_->field_ref();
  const FieldT& co2coRatio  = co2coRatio_ ->field_ref();
  const FieldT& co2GasifRHS = co2GasifRHS_->field_ref();
  const FieldT& h2oGasifRHS = h2oGasifRHS_->field_ref();

  SpatFldPtr<FieldT> coFrac = SpatialFieldStore::get<FieldT,FieldT>( totalMass );

  // fraction of oxidized char that ends up as CO. The balance ends up as CO2.
  *coFrac <<= 1. / ( 1. + co2coRatio );

  // contributions from oxidation and gasification
  *composition[CO2] <<= ( mwCO2_/mwChar_ ) * (1 -*coFrac) * charOxidRHS;

  *composition[CO ] <<= ( mwCO_/mwChar_ ) *
                     ( *coFrac * charOxidRHS
                       + 2. * co2GasifRHS
                       +      h2oGasifRHS );

  *composition[H2 ] <<= ( mwH2_/mwChar_ ) * h2oGasifRHS ;

  // contributions from evaporation
  *composition[H2O] <<= evapRHS;

  // contributions from devolatiliztion
  switch (devModel_) {
    case DEV::CPDM:
      assert( composition.find(H2 ) != composition.end() );
      assert( composition.find(CH4) != composition.end() );
      assert( composition.find(HCN) != composition.end() );
      assert( composition.find(NH3) != composition.end() );

      *composition[H2 ] <<= *composition[H2 ] + devSpecRHS_[CPD::H2 ]->field_ref();
      *composition[CO ] <<= *composition[CO ] + devSpecRHS_[CPD::CO ]->field_ref();
      *composition[CO2] <<= *composition[CO2] + devSpecRHS_[CPD::CO2]->field_ref();
      *composition[H2O] <<= *composition[H2O] + devSpecRHS_[CPD::H2O]->field_ref();
      *composition[CH4] <<= *composition[CH4] + devSpecRHS_[CPD::CH4]->field_ref();
      *composition[HCN] <<= *composition[HCN] + devSpecRHS_[CPD::HCN]->field_ref();
      *composition[NH3] <<= *composition[NH3] + devSpecRHS_[CPD::NH3]->field_ref();
      break;

    case DEV::KOBAYASHIM:
      *composition[CO ] <<= *composition[CO ] + devSpecRHS_[SAROFIM::CO ]->field_ref();
      *composition[H2 ] <<= *composition[H2 ] + devSpecRHS_[SAROFIM::H2 ]->field_ref();
      break;

    case DEV::DAE:
    case DEV::SINGLERATE:
      *composition[CO ] <<= *composition[CO ] + devSpecRHS_[SNGRATE::CO ]->field_ref();
      *composition[H2 ] <<= *composition[H2 ] + devSpecRHS_[SNGRATE::H2 ]->field_ref();
      break;

    case DEV::INVALID_DEVMODEL:
      throw std::invalid_argument( "Invalid devolatilization model in ProducedGasComposition::evaluate()" );
  }

  // calculate the sum of all contributions
  totalMass <<= 0.;
  for(const GasSpeciesName& specEnum : specEnums_ ){
    totalMass <<= totalMass + *composition[specEnum];
  }

  /* normalize individual species contributions by the total to obtain
   * mass fractions of gas produced at the coal particle. If the overall
   * production rate is zero, set composition to 100% CO2. This is done
   * so that an enthalpy calculation is possible.
   */
  for( const GasSpeciesName& specEnum : specEnums_ ){
    const int delta = (specEnum == CO2);

    *composition[specEnum] <<= cond( abs(totalMass) == 0., delta )
                                   ( *composition[specEnum] / totalMass );
  }
}
//--------------------------------------------------------------------

template< typename FieldT >
SpeciesTagMap
ProducedGasComposition<FieldT>::
Builder::
get_produced_species_tag_map( const std::string tagPrefix,
                              DEV::DevModel     devModel ){
  const std::vector<GasSpeciesName> speciesEnums = get_produced_species( devModel );

  SpeciesTagMap tagMap;

  for( const GasSpeciesName& spec : speciesEnums ){
    tagMap[spec] = Expr::Tag(tagPrefix + species_name(spec) , Expr::STATE_NONE);
  }

  return tagMap;
}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::TagList
ProducedGasComposition<FieldT>::
Builder::
get_produced_species_tags( const std::string tagPrefix,
                           DEV::DevModel     devModel ){
  SpeciesTagMap tagMap = get_produced_species_tag_map( tagPrefix, devModel );

  Expr::TagList tags;
  for( const auto p : tagMap ) tags.push_back( p.second );

  return tags;
}

//--------------------------------------------------------------------

template< typename FieldT >
ProducedGasComposition<FieldT>::
Builder::Builder( const std::string    tagPrefix,
                  const Expr::Tag&     totalMassSrcTag,
                  const Expr::TagList& devSpecRHSTags,
                  const Expr::Tag&     evapRHSTag,
                  const Expr::Tag&     charOxidRHSTag,
                  const Expr::Tag&     co2coRatioTag,
                  const Expr::Tag&     co2GasifRHSTag,
                  const Expr::Tag&     h2oGasifRHSTag,
                  const DEV::DevModel devModel )
  : ExpressionBuilder( Expr::tag_list( Builder::get_produced_species_tags( tagPrefix, devModel ),
                                       totalMassSrcTag )
                      ),
    devSpecRHSTags_( devSpecRHSTags ),
    evapRHSTag_    ( evapRHSTag     ),
    charOxidRHSTag_( charOxidRHSTag ),
    co2coRatioTag_ ( co2coRatioTag  ),
    co2GasifRHSTag_( co2GasifRHSTag ),
    h2oGasifRHSTag_( h2oGasifRHSTag ),
    devModel_      ( devModel       ),
    specTagMap_    ( Builder::get_produced_species_tag_map( tagPrefix, devModel ) )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ProducedGasComposition<FieldT>::
Builder::build() const
{
  return new ProducedGasComposition<FieldT>( devSpecRHSTags_, evapRHSTag_, charOxidRHSTag_, co2coRatioTag_,
                                             co2GasifRHSTag_, h2oGasifRHSTag_, devModel_, specTagMap_ );
}

} // namespace Coal
#endif // ProducedGasComposition_h

