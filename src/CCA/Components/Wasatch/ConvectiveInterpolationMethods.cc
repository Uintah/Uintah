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

#include <boost/bimap.hpp>

#include <stdexcept>
#include <ostream>

namespace Wasatch {

  typedef boost::bimap<std::string,ConvInterpMethods> ConvInterpStringMap;
  static ConvInterpStringMap validConvInterpStrings;

  void set_conv_interp_string_map()
  {
    if( !validConvInterpStrings.empty() ) return;
    typedef ConvInterpStringMap::left_value_type LVT;
    validConvInterpStrings.left.insert( LVT("CENTRAL" , CENTRAL ) );
    validConvInterpStrings.left.insert( LVT("UPWIND"  , UPWIND  ) );
    validConvInterpStrings.left.insert( LVT("SUPERBEE", SUPERBEE) );
    validConvInterpStrings.left.insert( LVT("CHARM"   , CHARM   ) );
    validConvInterpStrings.left.insert( LVT("KOREN"   , KOREN   ) );
    validConvInterpStrings.left.insert( LVT("MC"      , MC      ) );
    validConvInterpStrings.left.insert( LVT("OSPRE"   , OSPRE   ) );
    validConvInterpStrings.left.insert( LVT("SMART"   , SMART   ) );
    validConvInterpStrings.left.insert( LVT("VANLEER" , VANLEER ) );
    validConvInterpStrings.left.insert( LVT("HCUS"    , HCUS    ) );
    validConvInterpStrings.left.insert( LVT("MINMOD"  , MINMOD  ) );
    validConvInterpStrings.left.insert( LVT("HQUICK"  , HQUICK  ) );
  }

  //------------------------------------------------------------------

  ConvInterpMethods get_conv_interp_method( std::string key )
  {
    set_conv_interp_string_map();
    std::transform( key.begin(), key.end(), key.begin(), ::toupper );
    ConvInterpStringMap::left_const_iterator ii = validConvInterpStrings.left.find(key);
    if( ii == validConvInterpStrings.left.end() ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "No matching upwind method for '" << key << "'" << std::endl;
    }
    return ii->second;
  }

  std::string get_conv_interp_method( const ConvInterpMethods key )
  {
    set_conv_interp_string_map();
    return validConvInterpStrings.right.find(key)->second;
  }

} // namespace Wasatch
