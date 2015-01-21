
#include <CCA/Components/Models/test/MaterialProperties.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;

MaterialProperties::MaterialProperties()
{
}

MaterialProperties::~MaterialProperties()
{
}

void
MaterialProperties::parse(ProblemSpecP& p)
{
  ProblemSpecP params = p->findBlock("properties");
  if(!params)
    throw ProblemSetupException("Cannot find properties block", __FILE__, __LINE__);
  params->require("cp", Cp);
  params->require("molecularweight", molecularWeight);
}

void
MaterialProperties::outputProblemSpec(ProblemSpecP& ps)
{
  ps->appendElement("cp",Cp);
  ps->appendElement("molecularweight",molecularWeight);
}
