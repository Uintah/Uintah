#include "ConstantMeltTemp.h"

using namespace Uintah;
	 
// Construct a melt temp model.  
ConstantMeltTemp::ConstantMeltTemp(ProblemSpecP& )
{
}

// Construct a copy of a melt temp model.  
ConstantMeltTemp::ConstantMeltTemp(const ConstantMeltTemp* )
{
}

// Destructor of melt temp model.  
ConstantMeltTemp::~ConstantMeltTemp()
{
}
	 
// Compute the melt temp
double 
ConstantMeltTemp::computeMeltingTemp(const PlasticityState* state)
{
  return state->initialMeltTemp;
}

