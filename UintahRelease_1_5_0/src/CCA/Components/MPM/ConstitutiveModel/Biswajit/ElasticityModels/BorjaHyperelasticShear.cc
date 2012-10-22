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

#include "BorjaHyperelasticShear.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>


using namespace Uintah;
using namespace std;

// Construct a shear stress model.  
BorjaHyperelasticShear::BorjaHyperelasticShear(ProblemSpecP& ps )
{
  ps->require("mu_0", d_mu0);
  ps->require("p_0", d_p0);
  ps->require("epsv_0", d_epsv0);
  ps->require("alpha", d_alpha);
  ps->require("kappa_tilde", d_kappa);
}

// Construct a copy of a shear stress model.  
BorjaHyperelasticShear::BorjaHyperelasticShear(const BorjaHyperelasticShear* ssm)
{
  d_mu0 = ssm->d_mu0;
  d_p0 = ssm->d_p0;
  d_epsv0 = ssm->d_epsv0;
  d_alpha = ssm->d_alpha;
  d_kappa = ssm->d_kappa;
}

// Destructor of shear stress model.  
BorjaHyperelasticShear::~BorjaHyperelasticShear()
{
}

void BorjaHyperelasticShear::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_stress_model");
  shear_ps->setAttribute("type", "borja");
  shear_ps->appendElement("mu_0", d_mu0);
  shear_ps->appendElement("p_0", d_p0);
  shear_ps->appendElement("epsv_0", d_epsv0);
  shear_ps->appendElement("alpha", d_alpha);
  shear_ps->appendElement("kappa_tilde", d_kappa);
}

// Compute the shear stress (not an increment in this case)
//   sigma_shear = 2 mu dev[eps] = sqrt(2/3) q n 
// where
//   q = 3 mu eps_s ,   eps_s = sqrt{2/3} ||dev[eps]|| ,   n = dev[eps]/||dev[eps]||
// and
//   mu = mu_0 + alpha p_0 exp[(eps_v - eps_v0)/kappa_tilde]
void 
BorjaHyperelasticShear::computeShearStress(const DeformationState* state,
                                           Matrix3& shear_stress)
{
  state->computeAlmansiStrain();
  double mu = d_mu0 + d_alpha*d_p0*exp((state->eps_v - d_epsv0)/d_kappa);
  shear_stress = dev_strain*(2.0*mu);
}

