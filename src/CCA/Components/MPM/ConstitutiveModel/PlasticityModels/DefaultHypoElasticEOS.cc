/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



#include "DefaultHypoElasticEOS.h"
#include <cmath>

using namespace Uintah;
using namespace SCIRun;

DefaultHypoElasticEOS::DefaultHypoElasticEOS()
{
} 

DefaultHypoElasticEOS::DefaultHypoElasticEOS(ProblemSpecP&)
{
} 
         
DefaultHypoElasticEOS::DefaultHypoElasticEOS(const DefaultHypoElasticEOS*)
{
} 
         
DefaultHypoElasticEOS::~DefaultHypoElasticEOS()
{
}


void DefaultHypoElasticEOS::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","default_hypo");
}
         

//////////
// Calculate the pressure using the elastic constitutive equation
double 
DefaultHypoElasticEOS::computePressure(const MPMMaterial* ,
                                       const PlasticityState* state,
                                       const Matrix3& ,
                                       const Matrix3& rateOfDeformation,
                                       const double& delT)
{
  // Get the state data
  double kappa = state->bulkModulus;
  double p_n = state->pressure;

  // Calculate pressure increment
  double delp = rateOfDeformation.Trace()*(kappa*delT);

  // Calculate pressure
  double p = p_n + delp;
  return p;
}

double 
DefaultHypoElasticEOS::eval_dp_dJ(const MPMMaterial* matl,
                                  const double& detF, 
                                  const PlasticityState* state)
{
  return (state->bulkModulus/detF);
}
