
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{ namespace ArchesCore {

INTERPOLANT get_interpolant_from_string(const std::string value){

  if ( value == "second_central" ){
    return SECONDCENTRAL;
  } else if ( value == "fourth_central" ){
    return FOURTHCENTRAL;
  } else {
    throw InvalidValue("Error: interpolar type not recognized: "+value, __FILE__, __LINE__);
  }

}

}}
