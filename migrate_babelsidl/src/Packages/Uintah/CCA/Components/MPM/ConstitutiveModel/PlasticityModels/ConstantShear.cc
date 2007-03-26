#include "ConstantShear.h"

using namespace Uintah;
	 
// Construct a shear modulus model.  
ConstantShear::ConstantShear(ProblemSpecP& )
{
}

// Construct a copy of a shear modulus model.  
ConstantShear::ConstantShear(const ConstantShear* )
{
}

// Destructor of shear modulus model.  
ConstantShear::~ConstantShear()
{
}


void ConstantShear::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_modulus_model");
  shear_ps->setAttribute("type","constant_shear");
}
	 
// Compute the shear modulus
double 
ConstantShear::computeShearModulus(const PlasticityState* state)
{
  return state->initialShearModulus;
}

