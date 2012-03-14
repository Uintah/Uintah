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

#ifndef Wasatch_StringNames_h
#define Wasatch_StringNames_h

#include <string>

namespace Wasatch{

  /**
   *  \ingroup WasatchFields
   *  \ingroup WasatchCore
   *
   *  \class  StringNames
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Defines names for variables used in Wasatch.
   *
   *  Note: this class is implemented in a singleton.  Access it as follows:
   *  <code>const StringNames& sName = StringNames::self();</code>
   */
  class StringNames
  {
  public:

    /**
     *  Access the StringNames object.
     */
    static const StringNames& self();

    const std::string time;

    const std::string
      xsvolcoord,  ysvolcoord,  zsvolcoord,
      xxvolcoord,  yxvolcoord,  zxvolcoord,
      xyvolcoord,  yyvolcoord,  zyvolcoord,
      xzvolcoord,  yzvolcoord,  zzvolcoord;

    // energy related variables
    const std::string
      temperature,
      e0, rhoE0,
      enthalpy,
      xHeatFlux, yHeatFlux, zHeatFlux;

    // species related variables
    const std::string
      species,
      rhoyi,
      xSpeciesDiffFlux, ySpeciesDiffFlux, zSpeciesDiffFlux,
      mixtureFraction;

    // thermochemistry related variables
    const std::string
      heatCapacity,
      thermalConductivity,
      viscosity;

    // momentum related variables
    const std::string
      xvel, yvel, zvel,
      xmom, ymom, zmom,
      pressure,
      tauxx, tauxy, tauxz,
      tauyx, tauyy, tauyz,
      tauzx, tauzy, tauzz;

  private:
    StringNames();
  };

} // namespace Wasatch

#endif // Wasatch_StringNames_h
