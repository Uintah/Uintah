#include "MTSShear.h"
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
MTSShear::MTSShear(ProblemSpecP& ps )
{
  ps->require("mu_0",d_mu0);
  ps->require("D",d_D);
  ps->require("T_0",d_T0);
}

// Construct a copy of a shear modulus model.  
MTSShear::MTSShear(const MTSShear* smm)
{
  d_mu0 = smm->d_mu0;
  d_D = smm->d_D;
  d_T0 = smm->d_T0;
}

// Destructor of shear modulus model.  
MTSShear::~MTSShear()
{
}
	 
// Compute the shear modulus
double 
MTSShear::computeShearModulus(const PlasticityState* state)
{
  double T = state->temperature;
  ASSERT(T > 0.0);
  double expT0_T = exp(d_T0/T) - 1.0;
  ASSERT(expT0_T != 0);
  double mu = d_mu0 - d_D/expT0_T;
  if (!(mu > 0.0)) {
    ostringstream desc;
    desc << "**Compute MTS Shear Modulus ERROR** Shear modulus <= 0." << endl;
    desc << "T = " << T << " mu0 = " << d_mu0 << " T0 = " << d_T0
         << " exp(To/T) = " << expT0_T << " D = " << d_D << endl;
    throw InvalidValue(desc.str());
  }
  return mu;
}

