#include <CCA/Components/Wasatch/FieldAdaptor.h>

#include <map>
#include <string>
#include <algorithm>

namespace Wasatch{

  typedef std::map<std::string,FieldTypes> StringMap;
  static StringMap validStrings;

  void set_string_map()
  {
    if( !validStrings.empty() ) return;

    validStrings["SVOL"  ] = SVOL;
    validStrings["SSURFX"] = SSURFX;
    validStrings["SSURFY"] = SSURFY;
    validStrings["SSURFZ"] = SSURFZ;

    validStrings["XVOL"  ] = XVOL;
    validStrings["XSURFX"] = XSURFX;
    validStrings["XSURFY"] = XSURFY;
    validStrings["XSURFZ"] = XSURFZ;

    validStrings["YVOL"  ] = YVOL;
    validStrings["YSURFX"] = YSURFX;
    validStrings["YSURFY"] = YSURFY;
    validStrings["YSURFZ"] = YSURFZ;

    validStrings["ZVOL"  ] = ZVOL;
    validStrings["ZSURFX"] = ZSURFX;
    validStrings["ZSURFY"] = ZSURFY;
    validStrings["ZSURFZ"] = ZSURFZ;
  }

  //------------------------------------------------------------------

  FieldTypes get_field_type( std::string key )
  {
    set_string_map();
    std::transform( key.begin(), key.end(), key.begin(), ::toupper );
    return validStrings[key];
  }

  //------------------------------------------------------------------

} // namespace Wasatch
