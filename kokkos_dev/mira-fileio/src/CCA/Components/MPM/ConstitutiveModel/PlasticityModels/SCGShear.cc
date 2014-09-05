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

#include "SCGShear.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace std;

// Construct a shear modulus model.  
SCGShear::SCGShear(ProblemSpecP& ps )
{
  ps->require("mu_0",d_mu0);
  ps->require("A",d_A);
  ps->require("B",d_B);
}

// Construct a copy of a shear modulus model.  
SCGShear::SCGShear(const SCGShear* smm)
{
  d_mu0 = smm->d_mu0;
  d_A = smm->d_A;
  d_B = smm->d_B;
}

// Destructor of shear modulus model.  
SCGShear::~SCGShear()
{
}

void SCGShear::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_modulus_model");
  shear_ps->setAttribute("type","scg_shear");

  shear_ps->appendElement("mu_0",d_mu0);
  shear_ps->appendElement("A",d_A);
  shear_ps->appendElement("B",d_B);
}

         
// Compute the shear modulus
double 
SCGShear::computeShearModulus(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  ASSERT(eta > 0.0);
  eta = pow(eta, 1.0/3.0);

  // Pressure is +ve in this calcualtion
  double P = -state->pressure;
  double mu = d_mu0*(1.0 + d_A*P/eta - d_B*(state->temperature - 300.0));
  return mu;
}

