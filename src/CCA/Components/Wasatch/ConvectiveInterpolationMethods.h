/*
 * The MIT License
 *
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


#ifndef Wasatch_ConvectiveInterpolationMethods_h
#define Wasatch_ConvectiveInterpolationMethods_h

#include <map>
#include <string>
#include <algorithm>

/**
 * \file ConvectiveInterpolationMethods.h
 */

namespace Wasatch {

  /**
   *  \enum ConvInterpMethods
   *  \brief the supported flux limiters
   */
  enum ConvInterpMethods {
    CENTRAL, //!< CENTRAL
    UPWIND,  //!< UPWIND
    SUPERBEE,//!< SUPERBEE
    CHARM,   //!< CHARM
    KOREN,   //!< KOREN
    MC,      //!< MC
    OSPRE,   //!< OSPRE
    SMART,   //!< SMART
    VANLEER, //!< VANLEER
    HCUS,    //!< HCUS
    MINMOD,  //!< MINMOD
    HQUICK   //!< HQUICK
  };

  /**
   *  \ingroup WasatchParser
   *
   *  \brief Given the string name for the interpolation method, this
   *         returns the associated enum.
   */
  ConvInterpMethods get_conv_interp_method( std::string key );

  /**
   * \ingroup WasatchParser
   * \param method the ConvInterpMethods enum value
   * \return the corresponding string name
   */
  std::string get_conv_interp_method( const ConvInterpMethods );

} // namespace Wasatch

#endif
