/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

#include "NeoHookean.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>


using namespace Uintah;
using namespace std;

// Construct a shear stress model.  
NeoHookean::NeoHookean(ProblemSpecP& ps )
{
  ps->require("shear_modulus", d_mu);
}

// Construct a copy of a shear stress model.  
NeoHookean::NeoHookean(const NeoHookean* ssm)
{
  d_mu = ssm->d_mu;
}

// Destructor of shear stress model.  
NeoHookean::~NeoHookean()
{
}

void NeoHookean::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_stress_model");
  shear_ps->setAttribute("type", "neo_hookean");
  shear_ps->appendElement("shear_modulus", d_mu);
}

// Compute the shear stress (not an increment in this case)
//   sigma_shear = dev(B_bar)*(mu/J)
// where
//   B_bar = 1/J^(2/3) B = 1/J^(2/3) F.Ft
void 
NeoHookean::computeShearStress(const DeformationState* state,
                               Matrix3& shear_stress)
{
  state->computeCauchyGreenBbar();
  shear_stress = state->dev_strain*(d_mu/state->J);
}

