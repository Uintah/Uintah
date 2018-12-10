#include <CCA/Components/Arches/Transport/TransportHelper.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{ namespace ArchesCore {

ArchesCore::EQUATION_CLASS assign_eqn_class_enum( std::string my_class ){

  using namespace ArchesCore;

  if ( my_class == "density_weighted" ){
    return DENSITY_WEIGHTED;
  } else if ( my_class == "dqmom" ){
    return DQMOM;
  } else if ( my_class == "volumetric"){
    return NO_PREMULT;
  } else if ( my_class == "momentum" ){
    return MOMENTUM;
  } else {
    throw ProblemSetupException( "Error: eqn group type not recognized.",
                                 __FILE__, __LINE__ );
  }
}

std::string get_premultiplier_name(ArchesCore::EQUATION_CLASS eqn_class){

  using namespace ArchesCore;

  switch( eqn_class ){
    case DENSITY_WEIGHTED:
      return "rho_";
    case DQMOM:
      //return "w_";
      return "";
    case NO_PREMULT:
      return "none";
    default:
      return "none";
  }
}

std::string get_postmultiplier_name(ArchesCore::EQUATION_CLASS eqn_class){

  using namespace ArchesCore;

  switch( eqn_class ){
    case DENSITY_WEIGHTED:
      return "";
    case DQMOM:
      return "qn";
    case NO_PREMULT:
      return "none";
    default:
      return "none";
  }
}
}} //Uintah::ArchesCore
