#include "CharBase.h"

#include <stdexcept>
#include <sstream>

namespace CHAR{

  std::string char_model_name( const CharModel model )
  {
    std::string name;
    switch (model){
      case LH               : name = "LH";          break;
      case FRACTAL          : name = "FRACTAL";     break;
      case FIRST_ORDER      : name = "FIRST_ORDER"; break;
      case CCK              : name = "CCK";         break;
      case INVALID_CHARMODEL: name = "INVALID";     break;
    }
    return name;
  }

  CharModel char_model( const std::string& name )
  {
    if     ( name == char_model_name(LH         ) ) return LH;
    else if( name == char_model_name(FRACTAL    ) ) return FRACTAL;
    else if( name == char_model_name(FIRST_ORDER) ) return FIRST_ORDER;
    else if( name == char_model_name(CCK        ) ) return CCK;
    else{
      std::ostringstream msg;
      msg << std::endl
          << __FILE__ << " : " << __LINE__ << std::endl
          << "Unsupported char chemistry model: '" << name << "'\n\n"
          << " Supported models:"
          << "\n\t" << char_model_name( LH          )
          << "\n\t" << char_model_name( FRACTAL     )
          << "\n\t" << char_model_name( FIRST_ORDER )
          << "\n\t" << char_model_name( CCK         )
          << std::endl;
      throw std::invalid_argument( msg.str() );
    }
    return INVALID_CHARMODEL;
  }

}
