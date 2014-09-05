#include "ConstantCp.h"

using namespace Uintah;
	 
// Construct a specific heat model.  
ConstantCp::ConstantCp(ProblemSpecP& )
{
}

// Construct a copy of a specific heat model.  
ConstantCp::ConstantCp(const ConstantCp* )
{
}

// Destructor of specific heat model.  
ConstantCp::~ConstantCp()
{
}
	 
void ConstantCp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP cm_ps = ps->appendChild("specific_heat_model");
  cm_ps->setAttribute("type","constant_Cp");
}

// Compute the specific heat
double 
ConstantCp::computeSpecificHeat(const PlasticityState* state)
{
  return state->specificHeat;
}

