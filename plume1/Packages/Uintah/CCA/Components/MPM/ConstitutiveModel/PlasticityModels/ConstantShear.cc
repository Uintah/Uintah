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
	 
// Compute the shear modulus
double 
ConstantShear::computeShearModulus(const PlasticityState* state)
{
  return state->initialShearModulus;
}

