#include "ConvectiveInterpolationMethods.h"

namespace Wasatch {
  typedef std::map<std::string,ConvInterpMethods> ConvInterpStringMap;
  static ConvInterpStringMap validConvInterpStrings;
  
  void set_conv_interp_string_map()
  {
    if( !validConvInterpStrings.empty() ) return;
    
    validConvInterpStrings["CENTRAL" ] = CENTRAL;
    validConvInterpStrings["UPWIND"  ] = UPWIND;
    validConvInterpStrings["SUPERBEE"] = SUPERBEE;
    validConvInterpStrings["CHARM"   ] = CHARM;
    validConvInterpStrings["KOREN"   ] = KOREN;
    validConvInterpStrings["MC"      ] = MC;
    validConvInterpStrings["OSPRE"   ] = OSPRE;
    validConvInterpStrings["SMART"   ] = SMART;
    validConvInterpStrings["VANLEER" ] = VANLEER;
    validConvInterpStrings["HCUS"    ] = HCUS;
    validConvInterpStrings["MINMOD"  ] = MINMOD;
    validConvInterpStrings["HQUICK"  ] = HQUICK;
  }
  
  //------------------------------------------------------------------
  
  ConvInterpMethods get_conv_interp_method( std::string key )
  {
    set_conv_interp_string_map();
    std::transform( key.begin(), key.end(), key.begin(), ::toupper );
    return validConvInterpStrings[key];
  }
} // namespace Wasatch
