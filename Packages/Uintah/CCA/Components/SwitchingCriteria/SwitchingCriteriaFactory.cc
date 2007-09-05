#include <Packages/Uintah/CCA/Components/SwitchingCriteria/SwitchingCriteriaFactory.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/None.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/TimestepNumber.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/SimpleBurn.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/SteadyBurn.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/SteadyState.h>
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
    switch_criteria = scinew None();
  } else if (criteria == "timestep" || criteria == "Timestep" || 
             criteria == "TIMESTEP")  {
    switch_criteria = scinew TimestepNumber(switch_ps);
  } else if (criteria == "SimpleBurn" || criteria == "Simple_Burn" || 
             criteria == "simpleBurn" || criteria == "simple_Burn")  {
    switch_criteria = scinew SimpleBurnCriteria(switch_ps);
  } else if (criteria == "SteadyBurn" || criteria == "Steady_Burn" || 
             criteria == "steadyBurn" || criteria == "steady_Burn")  {
    switch_criteria = scinew SteadyBurnCriteria(switch_ps);
  } else if (criteria == "SteadyState" || criteria == "steadystate")  {
    switch_criteria = scinew SteadyState(switch_ps);
  } else {
    ostringstream warn;
    warn<<"\n ERROR:\n Unknown switching criteria (" << criteria << ")\n";
    throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
  }

  return switch_criteria;

}
