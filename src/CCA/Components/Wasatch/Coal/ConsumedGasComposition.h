#ifndef ConsumedGasComposition_h
#define ConsumedGasComposition_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/CoalData.h>

#include <cantera/IdealGasMix.h>
#include <pokitt/CanteraObjects.h> // include cantera wrapper


/**
 *   \class ConsumedGasComposition
 *   \Author Josh McConnell
 *   \Date   April 2018
 *
 *   \brief Calculates the species mass fractions and total consumption rate (kg/s)
 *          of gas produced at the particle. The rates inputed to this expression
 *          are "consumption" rates so all rate values should be negative. The
 *          the total consumption rate should always have a non-negative value.
 *
 *   \param o2RHSTag     : O2 consumption rate in gas ( kg/s)
 *   \param co2GasifTag  : char consumption due to co2 gasification reaction
 *   \param h2oGasifTag  : char consumption due to h2o gasification reaction
 *
 */
namespace Coal {

//--------------------------------------------------------------------

std::vector<GasSpeciesName>
get_consumed_species(){
  std::vector<GasSpeciesName> species;

  species.push_back( O2  ); // char oxidation
  species.push_back( H2O ); // H2O gasification
  species.push_back( CO2 ); // CO2 gasification

  return species;
}

//--------------------------------------------------------------------

  template< typename FieldT >
class ConsumedGasComposition
 : public Expr::Expression<FieldT>
{

  DECLARE_FIELDS( FieldT, o2RHS_, co2GasifRHS_, h2oGasifRHS_ )

  const std::vector<GasSpeciesName> specEnums_;

  const SpeciesTagMap& specTagMap_;

  Cantera::IdealGasMix* const gas_;

  const double mwChar_, mwH2O_, mwCO2_;

  const size_t nSpec_;

  ConsumedGasComposition( const Expr::Tag&     o2RHSTag,
                          const Expr::Tag&     co2GasifRHSTag,
                          const Expr::Tag&     h2oGasifRHSTag,
                          const SpeciesTagMap& specTagMap );
public:
  ~ConsumedGasComposition();

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *   \param tagPrefix        : prefix for species mass fraction tags set here
     *   \param totalMassSrcTag  : tag for total mass production rate. The corresponding
     *                             field should always have a non-negative value.
     *   \param o2RHSTag     : O2 consumption rate in gas ( kg/s)
     *   \param co2GasifTag  : char consumption due to co2 gasification reaction
     *   \param h2oGasifTag  : char consumption due to h2o gasification reaction
     */
    Builder( const std::string tagPrefix,
             const Expr::Tag&  totalMassSrcTag,
             const Expr::Tag&  o2RHSTag,
             const Expr::Tag&  co2GasifRHSTag,
             const Expr::Tag&  h2oGasifRHSTag );

    Expr::ExpressionBase* build() const;

    static SpeciesTagMap get_consumed_species_tag_map( const std::string tagPrefix );

    static Expr::TagList get_consumed_species_tags( const std::string tagPrefix );

  private:
    const Expr::Tag o2RHSTag_, co2GasifRHSTag_, h2oGasifRHSTag_;
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
  ConsumedGasComposition<FieldT>::
  ConsumedGasComposition( const Expr::Tag&     o2RHSTag,
                          const Expr::Tag&     co2GasifRHSTag,
                          const Expr::Tag&     h2oGasifRHSTag,
                          const SpeciesTagMap& specTagMap )
    : Expr::Expression<FieldT>(),
      specEnums_ ( get_consumed_species()                             ),
      specTagMap_( specTagMap                                         ),
      gas_       ( CanteraObjects::get_gasmix()                       ),
      mwChar_    ( gas_->atomicWeight   ( gas_->elementIndex("C"  ) ) ),
      mwH2O_     ( gas_->molecularWeight( gas_->speciesIndex("H2O") ) ),
      mwCO2_     ( gas_->molecularWeight( gas_->speciesIndex("CO2") ) ),
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

    o2RHS_      = this->template create_field_request<FieldT>( o2RHSTag       );
    co2GasifRHS_= this->template create_field_request<FieldT>( co2GasifRHSTag );
    h2oGasifRHS_= this->template create_field_request<FieldT>( h2oGasifRHSTag );

  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  ConsumedGasComposition<FieldT>::
  ~ConsumedGasComposition()
  {
    CanteraObjects::restore_gasmix(gas_);
  }

//--------------------------------------------------------------------

template< typename FieldT >
void
ConsumedGasComposition<FieldT>::
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

  FieldT& totalMass = *resultVec[nSpec_];

  const FieldT& o2RHS       = o2RHS_      ->field_ref();
  const FieldT& co2GasifRHS = co2GasifRHS_->field_ref();
  const FieldT& h2oGasifRHS = h2oGasifRHS_->field_ref();

  *composition[O2 ] <<= o2RHS;
  *composition[CO2] <<= -( mwCO2_/mwChar_ ) * co2GasifRHS;
  *composition[H2O] <<= -( mwH2O_/mwChar_ ) * h2oGasifRHS;

  // calculate the sum of all contributions
  totalMass <<= 0.;
  for(const GasSpeciesName& specEnum : specEnums_ ){
    totalMass <<= totalMass + *composition[specEnum];
  }

  /* normalize individual species contributions by the total to obtain
   * mass fractions of gas consumed at the coal particle. If the overall
   * consumption rate is zero, set composition to 100% CO2. This is done
   * so that an enthalpy calculation is possible.
   */
  for(const GasSpeciesName& specEnum : specEnums_ ){
    const int delta = (specEnum == CO2);

    *composition[specEnum] <<= cond( abs(totalMass) == 0., delta )
                                   ( *composition[specEnum] / totalMass );
  }
}
//--------------------------------------------------------------------

template< typename FieldT >
SpeciesTagMap
ConsumedGasComposition<FieldT>::
Builder::
get_consumed_species_tag_map( const std::string tagPrefix ){
  const std::vector<GasSpeciesName> speciesEnums = get_consumed_species();

  SpeciesTagMap tagMap;

  for( const GasSpeciesName& spec : speciesEnums ){
    tagMap[spec] = Expr::Tag(tagPrefix + species_name(spec) , Expr::STATE_NONE);
  }

  return tagMap;
}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::TagList
ConsumedGasComposition<FieldT>::
Builder::
get_consumed_species_tags( const std::string tagPrefix ){
  SpeciesTagMap tagMap = get_consumed_species_tag_map( tagPrefix );

  Expr::TagList tags;

  for( const auto p : tagMap ){
    tags.push_back( p.second );
  }

  return tags;
}

//--------------------------------------------------------------------

template< typename FieldT >
ConsumedGasComposition<FieldT>::
Builder::Builder( const std::string    tagPrefix,
                  const Expr::Tag&     totalMassSrcTag,
                  const Expr::Tag&     o2RHSTag,
                  const Expr::Tag&     co2GasifRHSTag,
                  const Expr::Tag&     h2oGasifRHSTag )
  : ExpressionBuilder( Expr::tag_list( Builder::get_consumed_species_tags( tagPrefix ),
                                       totalMassSrcTag )
                      ),
    o2RHSTag_      ( o2RHSTag       ),
    co2GasifRHSTag_( co2GasifRHSTag ),
    h2oGasifRHSTag_( h2oGasifRHSTag ),
    specTagMap_    ( Builder::get_consumed_species_tag_map( tagPrefix ) )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ConsumedGasComposition<FieldT>::
Builder::build() const
{
  return new ConsumedGasComposition<FieldT>( o2RHSTag_, co2GasifRHSTag_, h2oGasifRHSTag_,
                                             specTagMap_ );
}

} // namespace Coal
#endif // ConsumedGasComposition_h

