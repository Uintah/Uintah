#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/STThermalContact.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Core/Containers/StaticArray.h>
#include <vector>

using namespace Uintah;

STThermalContact::STThermalContact(ProblemSpecP&,SimulationStateP& d_sS,
                                   MPMLabel* Mlb,MPMFlags* MFlag)
{
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
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
    StaticArray<NCVariable<double> > thermalContactTemperatureRate(numMatls);
    vector<double> Cp(numMatls);
    // for Fracture (additional field)-----------------------------------------
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > GTemp(numMatls);
    StaticArray<NCVariable<double> > GthermalContactTemperatureRate(numMatls);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
  
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[dwi], lb->gMassLabel,        dwi, patch, Ghost::None,0);
      new_dw->get(gTemp[dwi], lb->gTemperatureLabel, dwi, patch, Ghost::None,0);
      new_dw->allocateAndPut(thermalContactTemperatureRate[dwi],
                            lb->gThermalContactTemperatureRateLabel,dwi,patch);
      thermalContactTemperatureRate[dwi].initialize(0.);
      Cp[m]=mpm_matl->getSpecificHeat();
      if (flag->d_fracture) {
        // for Fracture (for additional field)----------------------------------
        new_dw->get(Gmass[dwi],lb->GMassLabel,       dwi, patch, Ghost::None,0);
        new_dw->get(GTemp[dwi],lb->GTemperatureLabel,dwi, patch, Ghost::None,0);
        new_dw->allocateAndPut(GthermalContactTemperatureRate[dwi],
                           lb->GThermalContactTemperatureRateLabel,dwi,patch);
        GthermalContactTemperatureRate[dwi].initialize(0);
      }
      // -------------------------------------------------------------------
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      double numerator=0.0;
      double denominator=0.0;
      IntVector c = *iter;
      for(int m = 0; m < numMatls; m++) {
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int n = mpm_matl->getDWIndex();
        numerator   += (gTemp[n][c] * gmass[n][c]  * Cp[m]);
        denominator += (gmass[n][c]  * Cp[m]);
        if (flag->d_fracture) {
          numerator   += GTemp[n][c] * Gmass[n][c]  * Cp[m];
          denominator += Gmass[n][c]  * Cp[m];  // add in second field;
        }
      }
      
      double contactTemperature = numerator/denominator;

      for(int m = 0; m < numMatls; m++) {
        thermalContactTemperatureRate[m][c] =
                                      (contactTemperature - gTemp[m][c])/delT;
        if (flag->d_fracture){
          GthermalContactTemperatureRate[m][c] =
                                      (contactTemperature - GTemp[m][c])/delT;
        }
      }
    }
  }
}

void STThermalContact::initializeThermalContact(const Patch* /*patch*/,
                                                int /*vfindex*/,
                                                DataWarehouse* /*new_dw*/)
{
}

void STThermalContact::addComputesAndRequires(Task* t, const PatchSet*,
                                              const MaterialSet*) const
{
  t->requires(Task::OldDW, lb->delTLabel);  
  t->requires(Task::NewDW, lb->gMassLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureLabel, Ghost::None);
  t->computes(lb->gThermalContactTemperatureRateLabel);
  if (flag->d_fracture) {
    // for second field, for Fracture ---------------------------------
    t->requires(Task::NewDW, lb->GMassLabel,        Ghost::None);
    t->requires(Task::NewDW, lb->GTemperatureLabel, Ghost::None);
    t->computes(lb->GThermalContactTemperatureRateLabel);
  }
}
