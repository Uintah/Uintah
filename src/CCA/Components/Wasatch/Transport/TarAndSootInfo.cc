/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

//-- Wasatch includes --//
#include "TarAndSootInfo.h"

namespace WasatchCore{


  TarAndSootInfo::TarAndSootInfo() :

      // tar properties
      tarHydrogen( 8. ),                  // (mol H)/(mol tar)
      tarCarbon( 10. ),                   // (mol C)/(mol tar)
      tarMW( 128.17 ),                    // g/mol
      tarDiffusivity( 8.6e-6 ),           // m^2/s, diffusion coefficient of naphthalene at 303K
      tarHeatOfOxidation( -1.8131e+07 ),  // J/kg, calculated from Hess's law

      // Soot properties. These parameters result in a mean particle diameter of ~40nm

      sootDensity( 1950. ),               // kg/m^3 [1]
      cMin( 3.2016e+6 ),                  // (carbon atoms)/(incipient soot particle)
      sootHeatOfOxidation( -9.2027e+06 )  // J/kg
  {}

  //------------------------------------------------------------------

  const TarAndSootInfo&
  TarAndSootInfo::self()
  {
    static const TarAndSootInfo s;
    return s;
  }

  //------------------------------------------------------------------

}

/*
 * source for parameters:
 *  [1]    Brown, A. L.; Fletcher, T. H.
 *         "Modeling Soot Derived from Pulverized Coal"
 *         Energy & Fuels, 1998, 12, 745-757
 */
