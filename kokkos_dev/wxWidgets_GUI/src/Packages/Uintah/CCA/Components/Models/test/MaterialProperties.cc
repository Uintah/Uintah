
#include <Packages/Uintah/CCA/Components/Models/test/MaterialProperties.h>
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
    throw ProblemSetupException("Cannot find properties block", __FILE__, __LINE__);
  params->require("cp", Cp);
  params->require("molecularweight", molecularWeight);
}

void MaterialProperties::outputProblemSpec(ProblemSpecP& ps)
{
  ps->appendElement("cp",Cp,false,4);
  ps->appendElement("molecularweight",molecularWeight,false,4);

}
