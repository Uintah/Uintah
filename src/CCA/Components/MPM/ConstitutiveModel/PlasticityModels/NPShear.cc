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

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include "NPShear.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>


using namespace Uintah;
using namespace std;

// Construct a shear modulus model.  
NPShear::NPShear(ProblemSpecP& ps )
{
  ps->require("mu_0",d_mu0);
  ps->require("zeta",d_zeta);
  ps->require("slope_mu_p_over_mu0",d_slope_mu_p_over_mu0);
  ps->require("C",d_C);
  ps->require("m",d_m);
}

// Construct a copy of a shear modulus model.  
NPShear::NPShear(const NPShear* smm)
{
  d_mu0 = smm->d_mu0;
  d_zeta = smm->d_zeta;
  d_slope_mu_p_over_mu0 = smm->d_slope_mu_p_over_mu0;
  d_C = smm->d_C;
  d_m = smm->d_m;
}

// Destructor of shear modulus model.  
NPShear::~NPShear()
{
}

void NPShear::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_modulus_model");
  shear_ps->setAttribute("type","np_shear");

  shear_ps->appendElement("mu_0",d_mu0);
  shear_ps->appendElement("zeta",d_zeta);
  shear_ps->appendElement("slope_mu_p_over_mu0",d_slope_mu_p_over_mu0);
  shear_ps->appendElement("C",d_C);
  shear_ps->appendElement("m",d_m);
}

         
// Compute the shear modulus
double 
NPShear::computeShearModulus(const PlasticityState* state)
{
  double That = state->temperature/state->meltingTemp;
  if (That <= 0) return d_mu0;

  double mu = 1.0e-8; // Small value to keep the code from crashing
  if (That > 1.0+d_zeta) return mu;

  double j_denom = d_zeta*(1.0 - That/(1.0+d_zeta));
  double J = 1.0 + exp((That-1.0)/j_denom);
  if (!finite(J)) return mu;

  double eta = state->density/state->initialDensity;
  ASSERT(eta > 0.0);
  eta = pow(eta, 1.0/3.0);

  // Pressure is +ve in this calculation
  double P = -state->pressure;
  double t1 = d_mu0*(1.0 + d_slope_mu_p_over_mu0*P/eta);
  double t2 = 1.0 - That;
  double k_amu = 1.3806503e4/1.6605402;
  double t3 = state->density*k_amu*state->temperature/(d_C*d_m);
  mu = 1.0/J*(t1*t2 + t3);

  if (mu < 1.0e-8) {
    cout << "mu = " << mu << " T = " << state->temperature
         << " Tm = " << state->meltingTemp << " T/Tm = " << That
         << " J = " << J << " rho/rho_0 = " << eta 
         << " p = " << P << " t1 = " << t1 
         << " t2 = " << t2 << " t3 = " << t3 << endl;
    mu = 1.0e-8;
  }
  return mu;
}

