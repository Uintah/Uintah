#ifndef CoalData_h
#define CoalData_h

#include <string>
#include <map>

#include <expression/Tag.h>
#include <spatialops/Nebo.h>

namespace Coal{

  /**
   *  \ingroup Coal
   *  \enum CoalType
   *  \brief Each supported coal type.  See also coal_type_name()
   */
  enum CoalType {
    NorthDakota_Lignite          = 0,
    Gillette_Subbituminous       = 1,
    MontanaRosebud_Subbituminous = 2,
    Illinois_Bituminous          = 3,
    Kentucky_Bituminous          = 4,
    Pittsburgh_Bituminous        = 5,
    Pittsburgh_Shaddix           = 6,
    Russian_Bituminous           = 7,
    Black_Thunder                = 8,
    Shenmu                       = 9,
    Guizhou                      = 10,
    Utah_Bituminous              = 11,
    Utah_Skyline                 = 12,
    Utah_Skyline_Char_10atm      = 13,
    Utah_Skyline_Char_12_5atm    = 14,
    Utah_Skyline_Char_15atm      = 15,
    Highvale                     = 16,
    Highvale_Char                = 17,
    Eastern_Bituminous           = 18,
    Eastern_Bituminous_Char      = 19,
    Illinois_No_6                = 20,
    Illinois_No_6_Char           = 21,
    INVALID_COALTYPE             = 99
  };

  /**
   *  \ingroup Coal
   *  \enum GasSpeciesName
   *  \brief Enumerates all of the gas phase species that the coal model is aware of.
   */
  enum GasSpeciesName{
    O2,
    CO,
    CO2,
    H2,
    H2O,
    CH4,
    HCN,
    NH3,
    INVALID_SPECIES
  };


  /**
   *  \brief Obtain the enumeration for the given gas-phase species.
   *
   *  \param gasName the name of the gas-phase species
   *
   *  Note: if the requested name is not found, the INVALID_SPECIES
   *  enum value will be returned.
   */
  GasSpeciesName gas_name_to_enum( const std::string );

  /**
   *  \typedef SpeciesTagMap
   *  \brief Used in the CoalInterface, this provides a way of mapping
   *         species tags onto the species enum.
   */
  typedef std::map< GasSpeciesName, Expr::Tag >  SpeciesTagMap;

  /**
   *  \ingroup Coal
   *  \fn string coal_type_name( const CoalType )
   *  \brief obtain the name of the coal given its enum value
   */
  std::string coal_type_name( const CoalType );

  /**
   *  \ingroup Coal
   *  \fn CoalType coal_type( const std::string& )
   *  \brief obtain the type of the coal given its string name
   */
  CoalType coal_type( const std::string& );

  /**
   *  \ingroup Coal
   *  \fn string species_name( const GasSpeciesNames )
   *  \brief obtain the species name given the enum
   */
  std::string species_name( const GasSpeciesName );

  /**
   *  \ingroup Coal
   *  \brief Heat fraction absorbs by particle, (1-0.7) goes to gas phase. 
   */
  inline double absored_heat_fraction_particle() {return 0.7;};

  /**
   *  \ingroup Coal
   *  \class CoalComposition
   *  \brief Describes the raw coal composition
   */
  class CoalComposition {
  public:
    CoalComposition( const Coal::CoalType sel );
    const double get_C()          const{ return C_; }
    const double get_H()          const{ return H_; }
    const double get_N()          const{ return N_; }
    const double get_O()          const{ return O_; }
    const double get_moisture()   const{ return moisture_; } ///< moisture fraction of parent coal
    const double get_ash()        const{ return ash_;      } ///< ash fraction of the parent coal
    const double get_vm()         const{ return vm_;       } ///< volatile matter fraction of the parent coal
    const double get_fixed_c()    const{ return fixedc_;   } ///< fixed carbon fraction of the parent coal
    bool test_data();
    double get_c0() const;
    double get_tarMonoMW() const{ return tarMonoMW_; }
  protected:
    double C_,H_,N_,S_,O_, moisture_, ash_, vm_, fixedc_, tarMonoMW_;
  };

} // namespace Coal

#endif // CoalData_h
