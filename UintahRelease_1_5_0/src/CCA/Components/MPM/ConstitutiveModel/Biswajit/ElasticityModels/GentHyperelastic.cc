/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include "GentHyperelastic.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>


using namespace Uintah;
using namespace std;

// Construct a shear stress model.  
GentHyperelastic::GentHyperelastic(ProblemSpecP& ps )
{
  ps->require("shear_modulus", d_mu);
  double I1_max = 0.0;
  ps->require("I1_max", I1_max);
  d_Jm = I1_max - 3.0;
}

// Construct a copy of a shear stress model.  
GentHyperelastic::GentHyperelastic(const GentHyperelastic* ssm)
{
  d_mu = ssm->d_mu;
  d_Jm = ssm->d_Jm;
}

// Destructor of shear stress model.  
GentHyperelastic::~GentHyperelastic()
{
}

void GentHyperelastic::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_stress_model");
  shear_ps->setAttribute("type", "gent");
  shear_ps->appendElement("shear_modulus", d_mu);
  shear_ps->appendElement("I1_max", d_Jm + 3.0);
}

// Compute the shear stress (not an increment in this case)
//   sigma_shear = (mu Jm)/(Jm - I1 + 3) B
// where
//   B = F.Ft 
// and
//   I1 = Tr(B) 
void 
GentHyperelastic::computeShearStress(const DeformationState* state,
                               Matrix3& shear_stress)
{
  state->computeCauchyGreenB();
  shear_stress = state_strain*(d_mu*d_Jm)/(Jm - state->eps_v + 3.0);
}

