#include <CCA/Components/Arches/TurbulenceModels/DynamicSmagorinskyHelper.h>

using namespace Uintah::ArchesCore;

Uintah::ArchesCore::FILTER
Uintah::ArchesCore::get_filter_from_string( const std::string & value ){

  if (      value == "simpson" ){      return SIMPSON; }
  else if ( value == "three_points" ){ return THREEPOINTS; }
  else if ( value == "box" ){          return BOX; }
  else {
    throw InvalidValue("Error: Filter type not recognized: "+value, __FILE__, __LINE__);
  }

}
