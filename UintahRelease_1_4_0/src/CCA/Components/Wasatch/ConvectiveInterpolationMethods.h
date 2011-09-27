#ifndef Wasatch_ConvectiveInterpolationMethods_h
#define Wasatch_ConvectiveInterpolationMethods_h

#include <map>
#include <string>
#include <algorithm>

namespace Wasatch {
  
  /**
   *  \enum ConvInterpMethods
   *  \brief the supported flux limiters
   */
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

  /**
   *  \ingroup WasatchParser
   *
   *  \brief Given the string name for the interpolation method, this
   *         returns the associated enum.
   *
   *  \todo need to add exception handling for invalid arguments
   */  
  ConvInterpMethods get_conv_interp_method( std::string key );

} // namespace Wasatch

#endif
