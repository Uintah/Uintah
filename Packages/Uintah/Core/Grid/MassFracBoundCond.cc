#include <Packages/Uintah/Core/Grid/MassFracBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

MassFractionBoundCond::MassFractionBoundCond(ProblemSpecP& ps,
					     std::string& kind, 
                                              std::string& variableName):
  BoundCond<double>(kind)
{
  d_type = variableName;
  ps->require("value",d_value);
}

MassFractionBoundCond::~MassFractionBoundCond()
{
}


MassFractionBoundCond* MassFractionBoundCond::clone()
{
  return new MassFractionBoundCond(*this);
}
