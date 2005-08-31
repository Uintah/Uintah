#include <Packages/Uintah/CCA/Components/SwitchingCriteria/SwitchingCriteriaFactory.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/None.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/TimestepNumber.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/PBXTemperature.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>


#include <iostream>
#include <string>
using std::cerr;
using std::endl;
using std::string;

using namespace Uintah;

SwitchingCriteria* SwitchingCriteriaFactory::create(ProblemSpecP& ps,
                                                    const ProcessorGroup* world)
{
  string criteria("");
  ProblemSpecP switch_ps = ps->findBlock("SwitchCriteria");
  if (switch_ps) {
    map<string,string> attributes;
    switch_ps->getAttributes(attributes);
    criteria = attributes["type"];
  } else {
    return 0;
  }

  SwitchingCriteria* switch_criteria = 0;
  if (criteria == "none" || criteria == "None" || criteria == "NONE") {
    switch_criteria = new None(switch_ps);
  } else if (criteria == "timestep" || criteria == "Timestep" || 
             criteria == "TIMESTEP")  {
    switch_criteria = new TimestepNumber(switch_ps);
  } else if (criteria == "PBXTemperature" || criteria == "pbxtemperature" || 
             criteria == "PBX_temperature")  {
    switch_criteria = new PBXTemperature(switch_ps);
  } else {
    throw ProblemSetupException("Unknown switching criteria."
                                "Valid criteria: None",
                                __FILE__, __LINE__);
  }

  return switch_criteria;

}
