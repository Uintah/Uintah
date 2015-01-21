#include <Core/Grid/BoundaryConditions/MassFracBoundCond.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

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
  return scinew MassFractionBoundCond(*this);
}
