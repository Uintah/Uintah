
#include <Packages/Uintah/Core/Grid/MaterialProperties.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;

MaterialProperties::MaterialProperties()
{
}

MaterialProperties::~MaterialProperties()
{
}

void MaterialProperties::parse(ProblemSpecP& p)
{
  ProblemSpecP params = p->findBlock("properties");
  if(!params)
    throw ProblemSetupException("Cannot find properties block");
  params->require("cp", Cp);
  params->require("molecularweight", molecularWeight);
}
