#ifndef SpeciesData_h
#define SpeciesData_h

#include <string>
#include <map>

#include <expression/Tag.h>
#include <spatialops/Nebo.h>

namespace Cantera{ class IdealGasMix; }

namespace GasSpec{
  /*
   * \ enum GasSpecies
   * \brief Enumerates species considered in the gas-phase
   */
  enum GasSpecies {
    H,
    H2,
    H2O,
    CH4,
    CO,
    CO2,
    O2,
    HCN,
    NH3,
    NO2,
    INVALID_SPECIES
    /*
     * Note: INVALID_SPECIES needs to be the last enumeration as it is used
     *       to set the number of species.
     */
  };

  /**
   *  \fn string species_name( const GasSpeciesNames )
   *  \brief obtain the species name given its enum
   */
  std::string species_name( const GasSpecies );

  class SpeciesData {
  public:
    SpeciesData();

    ~SpeciesData();

    /**
     *  \fn string nSpecies()
     *  \brief obtain number of gas-phase species considered
     */
    const size_t nSpecies() const{ return numSpecies_; };

    /**
     *  \fn string species_name_to_enum( const std::string  )
     *  \brief obtain the enum for a species given its name
     */
    const GasSpecies species_name_to_enum( const std::string ) const;

    /**
     *  \fn string species_index( const GasSpeciesNames )
     *  \brief obtain the index of a species given its name
     */
    const unsigned int species_index( const std::string ) const;

    /**
     *  \fn string get_mw( const std::string )
     *  \brief obtain the molecular weight (g/mol) of a species given its name
     */
    const double get_mw( const GasSpecies spec ) const;

  protected:
    unsigned int numSpecies_;
    std::vector<std::string> specNames_;

  private:
    Cantera::IdealGasMix* const gas_;
  };
}// namespace GasSpec

#endif // SpeciesData_h
