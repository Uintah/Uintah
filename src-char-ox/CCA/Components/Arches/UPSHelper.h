#ifndef Uintah_Component_Arches_UPSHelper_h
#define Uintah_Component_Arches_UPSHelper_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>

namespace Uintah{ namespace ArchesCore {

  enum CFD_ROLE { UVELOCITY, VVELOCITY, WVELOCITY, CCUVELOCITY, CCVVELOCITY, CCWVELOCITY,PRESSURE, TEMPERATURE, ENTHALPY, DENSITY,
                  TOTAL_VISCOSITY };

  static inline CFD_ROLE role_string_to_enum( const std::string role ){

    if ( role == "uvelocity" ){
      return UVELOCITY;
    } else if ( role == "vvelocity" ){
      return VVELOCITY;
    } else if ( role == "wvelocity" ){
      return WVELOCITY;
    } else if ( role == "ccuvelocity" ){
      return CCUVELOCITY;
    } else if ( role == "ccvvelocity" ){
      return CCVVELOCITY;
    } else if ( role == "ccwvelocity" ){
      return CCWVELOCITY;
    } else if ( role == "pressure" ){
      return PRESSURE;
    } else if ( role == "temperature" ){
      return TEMPERATURE;
    } else if ( role == "enthalpy" ){
      return ENTHALPY;
    } else if ( role == "density" ){
      return DENSITY;
    } else if ( role == "total_viscosity" ){
      return TOTAL_VISCOSITY;
    } else {
      throw InvalidValue("Error: Cannot match role to CFD_ROLE enum. ", __FILE__, __LINE__ );
    }
  }

  static inline std::string role_enum_to_string( const CFD_ROLE role ){
    if ( role == UVELOCITY ){
      return "uvelocity";
    } else if ( role == VVELOCITY ){
      return "vvelocity";
    } else if ( role == WVELOCITY ){
      return "wvelocity";
    } else if ( role == CCUVELOCITY ){
      return "ccuvelocity";
    } else if ( role == CCVVELOCITY ){
      return "ccvvelocity";
    } else if ( role == CCWVELOCITY ){
      return "ccwvelocity";
    } else if ( role == PRESSURE ){
      return "pressure";
    } else if ( role == TEMPERATURE ){
      return "temperature";
    } else if ( role == ENTHALPY ){
      return "enthalpy";
    } else if ( role == DENSITY ){
      return "density";
    } else if ( role == TOTAL_VISCOSITY ){
      return "total_viscosity";
    } else {
      throw InvalidValue("Error: Role enum type not recognized.", __FILE__, __LINE__ );
    }
  }

  /** @brief Parse the VarID section in the UPS file for a specific CFD role **/
  static std::string parse_ups_for_role( CFD_ROLE role_enum, ProblemSpecP db, std::string mydefault="NotSet"  ){

    std::string role = role_enum_to_string( role_enum );

    ProblemSpecP db_varid = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("VarID");

    if ( db_varid ){
      for ( ProblemSpecP db_id = db_varid->findBlock("var"); db_id != nullptr; db_id = db_id->findNextBlock("var") ){

        std::string label="NotFound";
        std::string ups_role;

        db_id->getAttribute("label", label);
        db_id->getAttribute("role", ups_role);

        if ( ups_role == role ){
          return label;
        }
      }
    }

    return mydefault;
  }

}} // end Uintah::ArchesCore

#endif
