#include "TurbulenceParameters.h"

namespace Wasatch {

  void parse_turbulence_input(Uintah::ProblemSpecP turbulenceInputParams,
                              TurbulenceParameters& turbParams)
  {
    if (!turbulenceInputParams) return;
    
    // get the name of the turbulence model
    std::string turbulenceModelName;
    turbulenceInputParams->get("TurbulenceModel",turbulenceModelName);    
    if ( turbulenceModelName.compare("SMAGORINSKY") == 0   ) {
      turbParams.turbulenceModelName = SMAGORINSKY;
    } else if ( turbulenceModelName.compare("DYNAMIC") ==0 ) {    
      turbParams.turbulenceModelName = DYNAMIC;  
    } else if ( turbulenceModelName.compare("WALE")==0     ) {    
      turbParams.turbulenceModelName = WALE;
    } else {
      turbParams.turbulenceModelName = NONE;
    }
    
    // get the eddy viscosity constant
    turbulenceInputParams->getWithDefault("EddyViscosityConstant",turbParams.eddyViscosityConstant, 0.1);
    
    // get the kolmogorov scale
    turbulenceInputParams->getWithDefault("KolmogorovScale",turbParams.kolmogorovScale, 1e100);    
  }
  
}