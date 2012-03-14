/*
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
