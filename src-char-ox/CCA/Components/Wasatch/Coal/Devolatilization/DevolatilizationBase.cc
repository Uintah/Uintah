#include <CCA/Components/Wasatch/Coal/Devolatilization/DevolatilizationBase.h>

#include <stdexcept>
#include <sstream>

namespace DEV{

  std::string dev_model_name( const DevModel model )
  {
    std::string name;
    switch (model){
      case CPDM            : name="CPD";         break;
      case KOBAYASHIM      : name="KobSarofim";  break;
      case SINGLERATE      : name="SingleRate";  break;
      case DAE             : name="DAE";         break;
      case INVALID_DEVMODEL: name="INVALID";     break;
    }
    return name;
  }

  DevModel devol_model( const std::string& name )
  {
    if     ( name == dev_model_name( CPDM       ) ) return CPDM;
    else if( name == dev_model_name( KOBAYASHIM ) ) return KOBAYASHIM;
    else if( name == dev_model_name( SINGLERATE ) ) return SINGLERATE;
    else if( name == dev_model_name( DAE        ) ) return DAE;
    else{
      std::ostringstream msg;
      msg << std::endl
          << __FILE__ << " : " << __LINE__ << std::endl
          << "Unsupported devolatilization model: '" << name << "'\n\n"
          << " Supported models:"
          << "\n\t" << dev_model_name(CPDM      )
          << "\n\t" << dev_model_name(KOBAYASHIM)
          << "\n\t" << dev_model_name(SINGLERATE)
          << "\n\t" << dev_model_name(DAE       )
          << std::endl;
      throw std::invalid_argument( msg.str() );
    }
    return INVALID_DEVMODEL;
  }

}
