#ifndef Wasatch_ConvectiveInterpolationMethods_h
#define Wasatch_ConvectiveInterpolationMethods_h

#include <map>
#include <string>

namespace Wasatch {
  
  enum ConvInterpMethods {
    CENTRAL,
    UPWIND,
    SUPERBEE,
    CHARM,
    KOREN,
    MC,
    OSPRE,
    SMART,
    VANLEER,
    HCUS,
    MINMOD,
    HQUICK
  };
  
  ConvInterpMethods get_conv_interp_method ( std::string key );
} // namespace Wasatch
#endif