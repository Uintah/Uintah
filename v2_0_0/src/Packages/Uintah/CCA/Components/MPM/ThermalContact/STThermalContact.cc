#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/STThermalContact.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <vector>

using namespace Uintah;

#define FRACTURE
#undef FRACTURE

STThermalContact::STThermalContact(ProblemSpecP&,SimulationStateP& d_sS,
								 MPMLabel* Mlb)
{
  d_sharedState = d_sS;
  lb = Mlb;
}

STThermalContact::~STThermalContact()
{
}

void STThermalContact::computeHeatExchange(const ProcessorGroup*,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMatls = d_sharedState->getNumMPMMatls();

    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gTemp(numMatls);
    StaticArray<NCVariable<double> > thermalContactHeatExchangeRate(numMatls);
    vector<double> Cp(numMatls);
#ifdef FRACTURE 
    // for Fracture (additional field)-----------------------------------------
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > GTemp(numMatls);
    StaticArray<NCVariable<double> > GthermalContactHeatExchangeRate(numMatls);
#endif
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);
  
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[dwi], lb->gMassLabel,        dwi, patch, Ghost::None,0);
      new_dw->get(gTemp[dwi], lb->gTemperatureLabel, dwi, patch, Ghost::None,0);
      new_dw->allocateAndPut(thermalContactHeatExchangeRate[dwi],
                            lb->gThermalContactHeatExchangeRateLabel,dwi,patch);
      thermalContactHeatExchangeRate[dwi].initialize(0.);
      Cp[m]=mpm_matl->getSpecificHeat();
//      K[m] =mpm_matl->getThermalConductivity();
#ifdef FRACTURE
      // for Fracture (for additional field)----------------------------------
      new_dw->get(Gmass[dwi], lb->GMassLabel,        dwi, patch, Ghost::None,0);
      new_dw->get(GTemp[dwi], lb->GTemperatureLabel, dwi, patch, Ghost::None,0);
      new_dw->allocateAndPut(GthermalContactHeatExchangeRate[dwi],
                            lb->GThermalContactHeatExchangeRateLabel,dwi,patch);
      GthermalContactHeatExchangeRate[dwi].initialize(0);
      // -------------------------------------------------------------------
#endif
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      double numerator=0.0;
      double denominator=0.0;
      for(int m = 0; m < numMatls; m++) {
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int n = mpm_matl->getDWIndex();
#ifndef FRACTURE 
	numerator   += (gTemp[n][*iter] * gmass[n][*iter] * Cp[m]);
#else
	numerator   += (gTemp[n][*iter] * gmass[n][*iter] * Cp[m]
                       +GTemp[n][*iter] * Gmass[n][*iter] * Cp[m]); //add second
#endif
#ifndef FRACTURE 
	denominator += (gmass[n][*iter] * Cp[m]);
#else
	denominator += (gmass[n][*iter] * Cp[m]
                       +Gmass[n][*iter] * Cp[m]);  // add in second field;
#endif
      }
      
      double contactTemperature = numerator/denominator;
      for(int m = 0; m < numMatls; m++) {
#ifndef FRACTURE 
         thermalContactHeatExchangeRate[m][*iter] =
                           (contactTemperature - gTemp[m][*iter])/delT;
#else
         GthermalContactHeatExchangeRate[m][*iter] =
                           (contactTemperature - GTemp[m][*iter])/delT;
#endif
      }
    }
  }
}

void STThermalContact::initializeThermalContact(const Patch* /*patch*/,
					int /*vfindex*/,
					DataWarehouse* /*new_dw*/)
{
}

void STThermalContact::addComputesAndRequires(Task* t,
					    const PatchSet*,
					    const MaterialSet*) const
{
  t->requires(Task::OldDW, lb->delTLabel);  
  t->requires(Task::NewDW, lb->gMassLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureLabel, Ghost::None);
  t->computes(lb->gThermalContactHeatExchangeRateLabel);
#ifdef FRACTURE
  // for second field, for Fracture ---------------------------------
  t->requires(Task::NewDW, lb->GMassLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->GTemperatureLabel, Ghost::None);
  t->computes(lb->GThermalContactHeatExchangeRateLabel);
#endif
}
