/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include "MTSShear.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace std;

// Construct a shear modulus model.  
MTSShear::MTSShear(ProblemSpecP& ps )
{
  ps->require("mu_0",d_mu0);
  ps->require("D",d_D);
  ps->require("T_0",d_T0);
}

// Construct a copy of a shear modulus model.  
MTSShear::MTSShear(const MTSShear* smm)
{
  d_mu0 = smm->d_mu0;
  d_D = smm->d_D;
  d_T0 = smm->d_T0;
}

// Destructor of shear modulus model.  
MTSShear::~MTSShear()
{
}

void MTSShear::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_modulus_model");
  shear_ps->setAttribute("type","mts_shear");

  shear_ps->appendElement("mu_0",d_mu0);
  shear_ps->appendElement("D",d_D);
  shear_ps->appendElement("T_0",d_T0);
}

         
// Compute the shear modulus
double 
MTSShear::computeShearModulus(const PlasticityState* state)
{
  double T = state->temperature;
  ASSERT(T > 0.0);
  double expT0_T = exp(d_T0/T) - 1.0;
  ASSERT(expT0_T != 0);
  double mu = d_mu0 - d_D/expT0_T;
  if (!(mu > 0.0)) {
    ostringstream desc;
    desc << "**Compute MTS Shear Modulus ERROR** Shear modulus <= 0." << endl;
    desc << "T = " << T << " mu0 = " << d_mu0 << " T0 = " << d_T0
         << " exp(To/T) = " << expT0_T << " D = " << d_D << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }
  return mu;
}

