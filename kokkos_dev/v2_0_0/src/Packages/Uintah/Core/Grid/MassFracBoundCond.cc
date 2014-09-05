#include <Packages/Uintah/Core/Grid/MassFracBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

MassFractionBoundCond::MassFractionBoundCond(ProblemSpecP& ps,std::string& kind, 
                                              std::string& variableName):
  BoundCond<double>(kind)
{
  d_type = variableName;
  ps->require("value",d_value);
}

MassFractionBoundCond::~MassFractionBoundCond()
{
}

double MassFractionBoundCond::getValue() const
{
  return d_value;
}

