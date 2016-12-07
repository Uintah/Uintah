#include "SpeciesData.h"

#include <stdexcept>
#include <sstream>

#include <cantera/IdealGasMix.h>
#include <pokitt/CanteraObjects.h> // include cantera wrapper

using std::string;
using std::ostringstream;
using std::endl;

namespace GasSpec{
  //------------------------------------------------------------------
  string species_name( const GasSpecies spec )
  {
    string name="";
    switch( spec ){
      case H  : name = "H"  ; break;
      case H2 : name = "H2" ; break;
      case H2O: name = "H2O"; break;
      case CO : name = "CO" ; break;
      case CO2: name = "CO2"; break;
      case CH4: name = "CH4"; break;
      case O2 : name = "O2" ; break;
      case HCN: name = "HCN"; break;
      case NH3: name = "NH3"; break;
      case NO2: name = "NO2"; break;
      case INVALID_SPECIES: name=""; break;
    }
    return name;
  }

  //------------------------------------------------------------------
  SpeciesData::
  SpeciesData()
   : gas_( CanteraObjects::get_gasmix() )
  {
    numSpecies_= (int)INVALID_SPECIES;

    specNames_.clear();
    specNames_.push_back( "H"   );
    specNames_.push_back( "H2"  );
    specNames_.push_back( "H2O" );
    specNames_.push_back( "CO"  );
    specNames_.push_back( "CO2" );
    specNames_.push_back( "CH4" );
    specNames_.push_back( "O2"  );
    specNames_.push_back( "HCN" );
    specNames_.push_back( "NH3" );
    specNames_.push_back( "NO2" );

    if( specNames_.size() != numSpecies_ ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "Size of species name vector is not consistent with the \n"
          << "number of species"   << std::endl
          << "number of species: " << numSpecies_
          << std::endl
          << std::endl
          << "specNames.size() : " << specNames_.size()
          << std::endl;
      throw std::runtime_error( msg.str() );
    }
  }

  //------------------------------------------------------------------

  SpeciesData::
  ~SpeciesData(){
    CanteraObjects::restore_gasmix(gas_);
  }
  //------------------------------------------------------------------

  /**
   *  \fn GasSpecies species_name_to_enum( const GasSpeciesName )
   *  \brief Obtain the enumeration for the given gas-phase species.
   *
   *  \param gasName the name of the gas-phase species
   *
   *  Note: if the requested name is not found, the INVALID_SPECIES
   *  enum value will be returned.
   */
  const GasSpecies
  SpeciesData::species_name_to_enum( const string name ) const
  {
    if     ( name == "H"   ) return H  ;
    else if( name == "H2"  ) return H2 ;
    else if( name == "H2O" ) return H2O;
    else if( name == "CO"  ) return CO ;
    else if( name == "CO2" ) return CO2;
    else if( name == "CH4" ) return CH4;
    else if( name == "O2"  ) return O2 ;
    else if( name == "HCN" ) return HCN;
    else if( name == "NH3" ) return NH3;
    else if( name == "NO2" ) return NO2;
    else{
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "Invalid gas species name passed to SpeciesData::species_name_to_enum." << std::endl
          << "Below is a list of considered gas-phase species:"<<std::endl;
      for( size_t i = 0; i<specNames_.size(); ++i ){
        msg << specNames_[i]<<std::endl;
      }
     throw std::runtime_error( msg.str() );
    }
  }

  //------------------------------------------------------------------
  const unsigned int
  SpeciesData::species_index( const string name ) const{
    const GasSpecies gSpec = species_name_to_enum( name );
    return (int) gSpec;
  }

  //------------------------------------------------------------------
  const double
  SpeciesData::get_mw( const GasSpecies spec ) const{

    size_t specIndex;
    switch(spec){
      case GasSpecies::CO2:  specIndex = gas_->speciesIndex("CO2");  break;
      case GasSpecies::H2O:  specIndex = gas_->speciesIndex("H2O");  break;
      case GasSpecies::O2 :  specIndex = gas_->speciesIndex("O2" );  break;
      case GasSpecies::H2 :  specIndex = gas_->speciesIndex("H2" );  break;
      case GasSpecies::CO :  specIndex = gas_->speciesIndex("CO" );  break;
      case GasSpecies::CH4:  specIndex = gas_->speciesIndex("CH4");  break;
      case GasSpecies::HCN:  specIndex = gas_->speciesIndex("HCN");  break;
      case GasSpecies::NH3:  specIndex = gas_->speciesIndex("NH3");  break;
      case GasSpecies::NO2:  specIndex = gas_->speciesIndex("NO2");  break;
      case GasSpecies::H  :  specIndex = gas_->speciesIndex("H"  );  break;

      default:
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "Invalid species." << std::endl
            << std::endl;
        throw std::runtime_error( msg.str() );
    }
    return gas_->molecularWeights()[specIndex];

  }

}// namespace GasSpec
