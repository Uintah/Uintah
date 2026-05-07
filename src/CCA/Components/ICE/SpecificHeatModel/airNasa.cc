/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

// ------------------ Specific Heat Model for ideal air ----------------
// Written by James Karr April 2026
// Computes the specfic heat and specfic heat ratio for air (3.76mol N2 per mol O2)
// using the nasa 7 polynomial. Thermo data can be found on the gri website: 
// http://combustion.berkeley.edu/gri-mech/data/nasa_plnm.html
// Valid for temperatures 200K to 3500K


#include <CCA/Components/ICE/SpecificHeatModel/airNasa.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>

#include <cmath> 

using namespace Uintah;

airNasa::airNasa( ProblemSpecP& ps )
  : SpecificHeat( ps )
{
}

airNasa::~airNasa()
{
}

void airNasa::outputProblemSpec( ProblemSpecP& ice_ps )
{
  ProblemSpecP model_ps = ice_ps->appendChild("SpecificHeatModel");
  model_ps->setAttribute("type", "airNasa");
}

double airNasa::getGamma( double T )
{
  return computeGamma_internal( T );
}

double airNasa::getInternalEnergy( double T )
{
  return computeInternalEnergy_internal( T );
}

template<class CCVar>
void airNasa::computeSpecificHeat_impl( CellIterator&       iter,
                                        CCVar&              temp_CC,
                                        CCVariable<double>& cv )
{
  for ( ; !iter.done(); iter++ ) {
    IntVector c = *iter;
    cv[c] = computeCv_internal( temp_CC[c] );
  }
}

template void airNasa::computeSpecificHeat_impl<CCVariable<double>>(
    CellIterator&, CCVariable<double>&, CCVariable<double>& );
template void airNasa::computeSpecificHeat_impl<constCCVariable<double>>(
    CellIterator&, constCCVariable<double>&, CCVariable<double>& );

double airNasa::computeCv_internal( double T ) const
{
  //--------------Formulas----------------
  // Cv = R - Cp
  // Cp = R Cp/Ru
  // Cp/Ru = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4 (Nasa 7 polynomial)

  double T2 = T  * T;
  double T3 = T2 * T;
  double T4 = T3 * T;

  const std::vector<double>& aOxygen =
    (T > d_Tmid) ? d_aOxygenHighTemp : d_aOxygenLowTemp;

  const std::vector<double>& aNitrogen =
    (T > d_Tmid) ? d_aNitrogenHighTemp : d_aNitrogenLowTemp;
  
  // Nasa-7 Polynomials for specfic heat
  double cp_oxygen   = aOxygen[0]   + aOxygen[1] * T   + aOxygen[2] * T2  + aOxygen[3] * T3  + aOxygen[4] * T4;
  double cp_nitrogen = aNitrogen[0] + aNitrogen[1] * T + aNitrogen[2] * T2 + aNitrogen[3] * T3 + aNitrogen[4] * T4;

  double cp_molarAir = d_xOxygen * cp_oxygen + d_xNitrogen * cp_nitrogen;

  double cp = Rair * cp_molarAir;

  return cp - Rair;
}

double airNasa::computeGamma_internal( double T ) const
{
 //--------------Formulas----------------
  // gamma = Cp / Cv
  // Cv = Cp - R
  // Cp = R Cp/Ru
  // Cp/Ru = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4 (Nasa 7 polynomial)

  double T2 = T  * T;
  double T3 = T2 * T;
  double T4 = T3 * T;

  const std::vector<double>& aOxygen =
    (T > d_Tmid) ? d_aOxygenHighTemp : d_aOxygenLowTemp;

  const std::vector<double>& aNitrogen =
    (T > d_Tmid) ? d_aNitrogenHighTemp : d_aNitrogenLowTemp;
  
  // Nasa-7 Polynomials for specfic heat
  double cp_oxygen   = aOxygen[0]   + aOxygen[1] * T   + aOxygen[2] * T2  + aOxygen[3] * T3  + aOxygen[4] * T4;
  double cp_nitrogen = aNitrogen[0] + aNitrogen[1] * T + aNitrogen[2] * T2 + aNitrogen[3] * T3 + aNitrogen[4] * T4;

  double cp_molarAir = d_xOxygen * cp_oxygen + d_xNitrogen * cp_nitrogen;

  double cp = Rair * cp_molarAir;
  double cv = cp - Rair;

  return cp / cv;
}

double airNasa::computeInternalEnergy_internal( double T ) const
{
  (void)T;
  return 0.0;
}
