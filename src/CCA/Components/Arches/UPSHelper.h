#ifndef Uintah_Component_Arches_UPSHelper_h
#define Uintah_Component_Arches_UPSHelper_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{ namespace ArchesCore {

  enum CFD_ROLE { UVELOCITY_ROLE,
                  VVELOCITY_ROLE,
                  WVELOCITY_ROLE,
                CCUVELOCITY_ROLE,
                CCVVELOCITY_ROLE,
                CCWVELOCITY_ROLE,
                   PRESSURE_ROLE,
                TEMPERATURE_ROLE,
                   ENTHALPY_ROLE,
                    DENSITY_ROLE,
            TOTAL_VISCOSITY_ROLE };

  static inline CFD_ROLE role_string_to_enum( const std::string role ){

    if ( role == "uvelocity" ){
      return UVELOCITY_ROLE;
    } else if ( role == "vvelocity" ){
      return VVELOCITY_ROLE;
    } else if ( role == "wvelocity" ){
      return WVELOCITY_ROLE;
    } else if ( role == "ccuvelocity" ){
      return CCUVELOCITY_ROLE;
    } else if ( role == "ccvvelocity" ){
      return CCVVELOCITY_ROLE;
    } else if ( role == "ccwvelocity" ){
      return CCWVELOCITY_ROLE;
    } else if ( role == "pressure" ){
      return PRESSURE_ROLE;
    } else if ( role == "temperature" ){
      return TEMPERATURE_ROLE;
    } else if ( role == "enthalpy" ){
      return ENTHALPY_ROLE;
    } else if ( role == "density" ){
      return DENSITY_ROLE;
    } else if ( role == "total_viscosity" ){
      return TOTAL_VISCOSITY_ROLE;
    } else {
      throw InvalidValue("Error: Cannot match role to CFD_ROLE enum. ", __FILE__, __LINE__ );
    }
  }

  static inline std::string role_enum_to_string( const CFD_ROLE role ){
    if ( role == UVELOCITY_ROLE ){
      return "uvelocity";
    } else if ( role == VVELOCITY_ROLE ){
      return "vvelocity";
    } else if ( role == WVELOCITY_ROLE ){
      return "wvelocity";
    } else if ( role == CCUVELOCITY_ROLE ){
      return "ccuvelocity";
    } else if ( role == CCVVELOCITY_ROLE ){
      return "ccvvelocity";
    } else if ( role == CCWVELOCITY_ROLE ){
      return "ccwvelocity";
    } else if ( role == PRESSURE_ROLE ){
      return "pressure";
    } else if ( role == TEMPERATURE_ROLE ){
      return "temperature";
    } else if ( role == ENTHALPY_ROLE ){
      return "enthalpy";
    } else if ( role == DENSITY_ROLE ){
      return "density";
    } else if ( role == TOTAL_VISCOSITY_ROLE ){
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

  /** @brief Find a node with a specific attribute **/
  /** Use the *=> combination of characters to delineate between nodes. **/
  /** The * indicates the node name. **/
  /** Everything starts at the ARCHES node **/
  static ProblemSpecP inline find_node_with_att( ProblemSpecP& db, std::string start,
                                          std::string children_name,
                                          std::string att,
                                          std::string att_value ){

    ProblemSpecP db_arches = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");
    std::string delimiter = "=>";
    size_t pos = 0;
    std::vector<std::string> nodes;
    while ((pos = start.find(delimiter)) != std::string::npos) {
      std::string n = start.substr(0, pos);
      start.erase(0, pos+delimiter.length());
      nodes.push_back(n);
    }

    ProblemSpecP db_parent = db;
    for ( auto i = nodes.begin(); i != nodes.end(); i++){
      if ( db_parent->findBlock(*i)){
        db_parent = db_parent->findBlock(*i);
      } else{
        throw ProblemSetupException("Error: UPS node not found - "+*i, __FILE__, __LINE__);
      }
    }

    //once the parent is found, assume there are many children with the same
    // name. The requires that we search all potential children and
    // compare attributes to the one sought after.
    for ( ProblemSpecP db_child = db_parent->findBlock(children_name); db_child != nullptr;
          db_child = db_child->findNextBlock(children_name)){
      std::string found_att;
      db_child->getAttribute(att, found_att);
      if ( found_att == att_value ){
        return db_child;
      }
    }

    return nullptr;

  }

  // Defining default names for specific CFD variables.
  static std::string default_uVel_name{"uVelocity"};             // u-velocity, staggered
  static std::string default_vVel_name{"vVelocity"};             // v-velocity, staggered
  static std::string default_wVel_name{"wVelocity"};             // w-velocity, staggered
  static std::string default_uMom_name{"x-mom"};                 // x-momentum, staggered
  static std::string default_vMom_name{"y-mom"};                 // y-momentum, staggered
  static std::string default_wMom_name{"z-mom"};                 // z-momentum, staggered
  static std::string default_viscosity_name{"total_viscosity"};  // total viscosity (molecular + turb closure) - note: turb closure may or may not exist

}} // end Uintah::ArchesCore

#endif
