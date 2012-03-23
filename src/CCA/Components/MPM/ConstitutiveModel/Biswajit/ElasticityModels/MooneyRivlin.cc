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

#include "MooneyRivlin.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>
#include <sstream>


using namespace Uintah;
using namespace std;

// Construct a shear stress model.  
MooneyRivlin::MooneyRivlin(ProblemSpecP& ps )
{
  ps->require("C_10", d_C1);
  ps->require("C_01", d_C2);
}

// Construct a copy of a shear stress model.  
MooneyRivlin::MooneyRivlin(const MooneyRivlin* ssm)
{
  d_C1 = ssm->d_C1;
  d_C2 = ssm->d_C2;
}

// Destructor of shear stress model.  
MooneyRivlin::~MooneyRivlin()
{
}

void MooneyRivlin::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_stress_model");
  shear_ps->setAttribute("type", "mooney_rivlin");
  shear_ps->appendElement("C_10", d_C1);
  shear_ps->appendElement("C_01", d_C2);
}

// Compute the shear stress (not an increment in this case)
//   sigma_shear = 2(C_10 + I1_bar*C_01) B_bar - 2 C_01 B_bar.B_bar
// where
//   B_bar = 1/J^(2/3) B = 1/J^(2/3) F.Ft
// and
//   I1_bar = 1/J^{2/3} I1 = 1/J^{2/3} Tr(B) = Tr(B_bar)
void 
MooneyRivlin::computeShearStress(const DeformationState* state,
                               Matrix3& shear_stress)
{
  state->computeCauchyGreenBbar();
  shear_stress = (state->strain*(d_C1 + state->eps_v*d_C2) - (state->strain*state->strain)*d_C2)*2.0;
}

