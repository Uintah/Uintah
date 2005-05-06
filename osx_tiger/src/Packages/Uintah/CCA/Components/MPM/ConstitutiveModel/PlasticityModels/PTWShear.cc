#include "PTWShear.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

// Construct a shear modulus model.  
PTWShear::PTWShear(ProblemSpecP& ps )
{
  ps->require("mu_0",d_mu0);
  ps->require("alpha",d_alpha);
  ps->require("alphap",d_alphap);
  ps->require("slope_mu_p_over_mu0",d_slope_mu_p_over_mu0);
}

// Construct a copy of a shear modulus model.  
PTWShear::PTWShear(const PTWShear* smm)
{
  d_mu0 = smm->d_mu0;
  d_alpha = smm->d_alpha;
  d_alphap = smm->d_alphap;
  d_slope_mu_p_over_mu0 = smm->d_slope_mu_p_over_mu0;
}

// Destructor of shear modulus model.  
PTWShear::~PTWShear()
{
}
	 
// Compute the shear modulus
double 
PTWShear::computeShearModulus(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  ASSERT(eta > 0.0);
  eta = pow(eta, 1.0/3.0);
  double That = state->temperature/state->meltingTemp;
  double mu0P = d_mu0*(1.0 + d_slope_mu_p_over_mu0*state->pressure/eta);
  double mu = mu0P*(1.0 - d_alphap*That);
  return mu;
}

