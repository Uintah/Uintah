#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/NullThermalContact.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Core/Util/NotFinished.h>

#include <vector>

using namespace Uintah;

#define FRACTURE
#undef FRACTURE

NullThermalContact::NullThermalContact(ProblemSpecP&,SimulationStateP& d_sS,
								MPMLabel* Mlb)
{
  d_sharedState = d_sS;
  lb = Mlb;
}

NullThermalContact::~NullThermalContact()
{
}

void NullThermalContact::computeHeatExchange(const ProcessorGroup*,
					     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse* ,
					     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();

      NCVariable<double> thermalContactHeatExchangeRate;
      new_dw->allocateAndPut(thermalContactHeatExchangeRate, 
			     lb->gThermalContactHeatExchangeRateLabel, 
			     dwindex, patch);

      thermalContactHeatExchangeRate.initialize(0);
#ifdef FRACTURE
      NCVariable<double> GthermalContactHeatExchangeRate;
      new_dw->allocateAndPut(GthermalContactHeatExchangeRate,
			     lb->GThermalContactHeatExchangeRateLabel, 
			     dwindex, patch);
      GthermalContactHeatExchangeRate.initialize(0);
#endif
    }
  }
}

void NullThermalContact::initializeThermalContact(const Patch* /*patch*/,
					int /*vfindex*/,
					DataWarehouse* /*new_dw*/)
{
}

void NullThermalContact::addComputesAndRequires(Task* t,
					    const PatchSet*,
					    const MaterialSet*) const
{
  t->computes(lb->gThermalContactHeatExchangeRateLabel);
#ifdef FRACTURE
  t->computes(lb->GThermalContactHeatExchangeRateLabel); 
#endif
}
