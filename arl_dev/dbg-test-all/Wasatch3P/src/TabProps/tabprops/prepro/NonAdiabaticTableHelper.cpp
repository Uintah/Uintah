/*
 * Copyright (c) 2014 The University of Utah
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

#include <cantera/Cantera.h>
#include <cantera/kernel/ThermoPhase.h>

#include <tabprops/prepro/rxnmdl/MixtureFraction.h>
#include <tabprops/prepro/NonAdiabaticTableHelper.h>

//====================================================================

AdEnthEvaluator::AdEnthEvaluator( MixtureFraction & mixfrac,
                                  const double fuelEnthalpy,
                                  const double oxidEnthalpy )
  : StateVarEvaluator( AD_ENTH, "AdiabaticEnthalpy" ),
    mixfrac_ ( mixfrac ),
    fuelEnth_( fuelEnthalpy ),
    oxidEnth_( oxidEnthalpy )
{}
//--------------------------------------------------------------------
double
AdEnthEvaluator::evaluate( const double & t,
                           const double & p,
                           const std::vector<double> & ys )
{
  // calculate the mixture fraction
  double f;
  mixfrac_.species_to_mixfrac( ys, f );

  // calculate the mixture enthalpy at adiabatic conditions - pure stream mixing.
  const double ha = f*fuelEnth_ + (1.0-f)*oxidEnth_;

#ifdef CGS_UNITS
  return ha * 1.0e4;
#else
  return ha;
#endif
}
//====================================================================
