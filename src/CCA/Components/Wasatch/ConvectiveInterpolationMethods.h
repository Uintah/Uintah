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
