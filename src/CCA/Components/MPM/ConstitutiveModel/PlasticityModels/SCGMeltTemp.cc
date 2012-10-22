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

#include "SCGMeltTemp.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace std;

// Construct a melt temp model.  
SCGMeltTemp::SCGMeltTemp(ProblemSpecP& ps )
{
  ps->require("Gamma_0",d_Gamma0);
  ps->require("a",d_a);
  ps->require("T_m0",d_Tm0);
}

// Construct a copy of a melt temp model.  
SCGMeltTemp::SCGMeltTemp(const SCGMeltTemp* mtm)
{
  d_Gamma0 = mtm->d_Gamma0;
  d_a = mtm->d_a;
  d_Tm0 = mtm->d_Tm0;
}

// Destructor of melt temp model.  
SCGMeltTemp::~SCGMeltTemp()
{
}

void SCGMeltTemp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP temp_ps = ps->appendChild("melting_temp_model");
  temp_ps->setAttribute("type","scg_Tm");

  temp_ps->appendElement("Gamma_0",d_Gamma0);
  temp_ps->appendElement("a",d_a);
  temp_ps->appendElement("T_m0",d_Tm0);
}

         
// Compute the melt temp
double 
SCGMeltTemp::computeMeltingTemp(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  double power = 2.0*(d_Gamma0 - d_a - 1.0/3.0);
  double Tm = d_Tm0*exp(2.0*d_a*(1.0 - 1.0/eta))*pow(eta,power);
  
  //cout << " SCG Melting Temp : " << Tm << " eta = " << eta << endl;
  return Tm;
}

