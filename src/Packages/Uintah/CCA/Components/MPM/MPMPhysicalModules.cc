
//Physical Models Interested:
#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>

#include "MPMPhysicalModules.h"

using namespace Uintah;

Contact*         MPMPhysicalModules::contactModel;
ThermalContact*  MPMPhysicalModules::thermalContactModel;

void MPMPhysicalModules::build(const ProblemSpecP& prob_spec,
                              SimulationStateP& sharedState)
{
   //solid mechanical contact
   contactModel = ContactFactory::create(prob_spec,sharedState);

   //solid thermal contact
   thermalContactModel = ThermalContactFactory::create(prob_spec, sharedState);
}

void MPMPhysicalModules::kill()
{
//  delete contactModel;
  delete thermalContactModel;
}
