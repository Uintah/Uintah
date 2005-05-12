#include "SCGShear.h"
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

