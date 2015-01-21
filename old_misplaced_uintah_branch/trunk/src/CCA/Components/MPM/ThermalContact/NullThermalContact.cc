#include <CCA/Components/MPM/ThermalContact/NullThermalContact.h>
#include <Core/Malloc/Allocator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Labels/MPMLabel.h>

#include <vector>

using namespace Uintah;

NullThermalContact::NullThermalContact(ProblemSpecP&,SimulationStateP& d_sS,
				       MPMLabel* Mlb,MPMFlags* MFlag)
{
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
}

NullThermalContact::~NullThermalContact()
{
}

void NullThermalContact::outputProblemSpec(ProblemSpecP& ps)
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

      NCVariable<double> thermalContactTemperatureRate;
      new_dw->allocateAndPut(thermalContactTemperatureRate, 
			     lb->gThermalContactTemperatureRateLabel, 
			     dwindex, patch);

      thermalContactTemperatureRate.initialize(0);
      NCVariable<double> GthermalContactTemperatureRate;
      if (flag->d_fracture) {
	new_dw->allocateAndPut(GthermalContactTemperatureRate,
			       lb->GThermalContactTemperatureRateLabel, 
			       dwindex, patch);
	GthermalContactTemperatureRate.initialize(0);
      }

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
  t->computes(lb->gThermalContactTemperatureRateLabel);
  if (flag->d_fracture)
    t->computes(lb->GThermalContactTemperatureRateLabel); 
}
