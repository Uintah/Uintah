#include <CCA/Components/Arches/UPSHelper.h>

// ---------------------------------------------------------------------------

using namespace Uintah;

std::string
ArchesCore::parse_ups_for_role( ArchesCore::CFD_ROLE role_enum, ProblemSpecP db, std::string mydefault  ){

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

// ---------------------------------------------------------------------------

ArchesCore::CFD_ROLE
ArchesCore::role_string_to_enum( const std::string role ){

  if ( role == "uvelocity" ){
    return ArchesCore::UVELOCITY_ROLE;
  } else if ( role == "vvelocity" ){
    return ArchesCore::VVELOCITY_ROLE;
  } else if ( role == "wvelocity" ){
    return ArchesCore::WVELOCITY_ROLE;
  } else if ( role == "ccuvelocity" ){
    return ArchesCore::CCUVELOCITY_ROLE;
  } else if ( role == "ccvvelocity" ){
    return ArchesCore::CCVVELOCITY_ROLE;
  } else if ( role == "ccwvelocity" ){
    return ArchesCore::CCWVELOCITY_ROLE;
  } else if ( role == "pressure" ){
    return ArchesCore::PRESSURE_ROLE;
  } else if ( role == "temperature" ){
    return ArchesCore::TEMPERATURE_ROLE;
  } else if ( role == "enthalpy" ){
    return ArchesCore::ENTHALPY_ROLE;
  } else if ( role == "density" ){
    return ArchesCore::DENSITY_ROLE;
  } else if ( role == "total_viscosity" ){
    return ArchesCore::TOTAL_VISCOSITY_ROLE;
  } else {
    throw InvalidValue("Error: Cannot match role to CFD_ROLE enum. ", __FILE__, __LINE__ );
  }
}

// ---------------------------------------------------------------------------

std::string
ArchesCore::role_enum_to_string( const ArchesCore::CFD_ROLE role ){
  if ( role == ArchesCore::UVELOCITY_ROLE ){
    return "uvelocity";
  } else if ( role == ArchesCore::VVELOCITY_ROLE ){
    return "vvelocity";
  } else if ( role == ArchesCore::WVELOCITY_ROLE ){
    return "wvelocity";
  } else if ( role == ArchesCore::CCUVELOCITY_ROLE ){
    return "ccuvelocity";
  } else if ( role == ArchesCore::CCVVELOCITY_ROLE ){
    return "ccvvelocity";
  } else if ( role == ArchesCore::CCWVELOCITY_ROLE ){
    return "ccwvelocity";
  } else if ( role == ArchesCore::PRESSURE_ROLE ){
    return "pressure";
  } else if ( role == ArchesCore::TEMPERATURE_ROLE ){
    return "temperature";
  } else if ( role == ArchesCore::ENTHALPY_ROLE ){
    return "enthalpy";
  } else if ( role == ArchesCore::DENSITY_ROLE ){
    return "density";
  } else if ( role == ArchesCore::TOTAL_VISCOSITY_ROLE ){
    return "total_viscosity";
  } else {
    throw InvalidValue("Error: Role enum type not recognized.", __FILE__, __LINE__ );
  }
}

ProblemSpecP
ArchesCore::find_node_with_att( ProblemSpecP& db, std::string start,
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

std::vector<bool>
ArchesCore::save_in_archiver( const std::vector<std::string> variables,
                              ProblemSpecP& db,
                              bool partial_match,
                              const int starting_pos_in,
                              const int starting_pos_lab )
{

  ProblemSpecP db_archive = db->getRootNode()->findBlock("DataArchiver");
  std::vector<bool> saved;
  int i = 0;

  if (!partial_match){

    // Searching for an exact match:
    for ( auto i_var = variables.begin(); i_var != variables.end(); i_var ++ ){ //loop over user variables
      saved.push_back(false);
      for ( ProblemSpecP db_child = db_archive->findBlock("save"); db_child != nullptr; //loop over dataArchiver
            db_child = db_child->findNextBlock("save") ){

        std::string label;
        db_child->getAttribute("label", label);

        if ( (*i_var).compare(label) == 0 ){
          saved[i] = true;
          break;
        }

      }
      i += 1;
    }

  } else {

    // Searching for a partial match:
    for ( auto i_var = variables.begin(); i_var != variables.end(); i_var ++ ){ //loop over user variables

      saved.push_back(false);
      std::string the_var = *i_var;

      for ( ProblemSpecP db_child = db_archive->findBlock("save"); db_child != nullptr; //loop over dataArchiver
            db_child = db_child->findNextBlock("save") ){

        std::string label;
        db_child->getAttribute("label", label);

        int size_of_var = (*i_var).size();
        int size_of_label = label.size();

        if ( size_of_label >= size_of_var ){
          //Can only compare the strings if the size of the label = or is > than the size of the var
          if ( the_var.compare( starting_pos_in, size_of_var, label, starting_pos_lab, size_of_var ) == 0 ){
            saved[i] = true;
            break;
          }
        }

      }

      i += 1;

    }
  }

  return saved;

}
