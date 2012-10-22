/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include "SteelCp.h"
#include <cmath>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
         
// Construct a specific heat model.  
SteelCp::SteelCp(ProblemSpecP& ps)
{
  d_Tc = 1040.0;
  ps->get("T_transition",d_Tc);
  d_A0 = 190.14;
  ps->get("A_LowT",d_A0);
  d_B0 = 273.75;
  ps->get("B_LowT",d_B0);
  d_C0 = 418.30;
  ps->get("C_LowT",d_C0);
  d_n0 = 0.2;
  ps->get("n_LowT",d_n0);
  d_A1 = 465.21;
  ps->get("A_HighT",d_A1);
  d_B1 = 267.52;
  ps->get("B_HighT",d_B1);
  d_C1 = 58.16;
  ps->get("C_HighT",d_C1);
  d_n1 = 0.35;
  ps->get("n_HighT",d_n1);
}

// Construct a copy of a specific heat model.  
SteelCp::SteelCp(const SteelCp* ccp)
{
  d_Tc = ccp->d_Tc;
  d_A0 = ccp->d_A0;
  d_B0 = ccp->d_B0;
  d_C0 = ccp->d_C0;
  d_n0 = ccp->d_n0;
  d_A1 = ccp->d_A1;
  d_B1 = ccp->d_B1;
  d_C1 = ccp->d_C1;
  d_n1 = ccp->d_n1;
}

// Destructor of specific heat model.  
SteelCp::~SteelCp()
{
}
         
void SteelCp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP cm_ps = ps->appendChild("specific_heat_model");
  cm_ps->setAttribute("type","steel_Cp");

  cm_ps->appendElement("T_transition",d_Tc);
  cm_ps->appendElement("A_LowT",d_A0);
  cm_ps->appendElement("B_LowT",d_B0);
  cm_ps->appendElement("C_LowT",d_C0);
  cm_ps->appendElement("n_LowT",d_n0);
  cm_ps->appendElement("A_HighT",d_A1);
  cm_ps->appendElement("B_HighT",d_B1);
  cm_ps->appendElement("C_HighT",d_C1);
  cm_ps->appendElement("n_HighT",d_n1);
}

// Compute the specific heat
double 
SteelCp::computeSpecificHeat(const PlasticityState* state)
{
  double Cp = 470.0;
  double T = state->temperature;

  // Specific heat model for 4340 steel (SI units)
  if (T == d_Tc) {
    T = T - 1.0;
  }
  if (T < d_Tc) {
    double t = 1 - T/d_Tc;
    Cp = d_A0 - d_B0*t + d_C0/pow(t, d_n0);
  } else {
    double t = T/d_Tc - 1.0;
    Cp = d_A1 + d_B1*t + d_C1/pow(t, d_n1);
  }
  return Cp;
}

