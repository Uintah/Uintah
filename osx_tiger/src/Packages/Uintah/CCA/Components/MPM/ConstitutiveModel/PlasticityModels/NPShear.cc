#include "NPShear.h"
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
	 
// Compute the shear modulus
double 
NPShear::computeShearModulus(const PlasticityState* state)
{
  double That = state->temperature/state->meltingTemp;
  ASSERT(That > 0.0);
  ASSERT(That < 1.0+d_zeta);
  double j_denom = d_zeta*(1.0 - That/(1.0+d_zeta));
  double J = 1.0 + exp((That-1.0)/j_denom);

  double eta = state->density/state->initialDensity;
  ASSERT(eta > 0.0);
  eta = pow(eta, 1.0/3.0);

  double t1 = d_mu0*(1.0 + d_slope_mu_p_over_mu0*state->pressure/eta);
  double t2 = 1.0 - That;
  double k_amu = 1.38e4/1.6605402;
  double t3 = state->density*k_amu*state->temperature/(d_C*d_m);
  double mu = 1.0/J*(t1*t2 + t3);
  return mu;
}

