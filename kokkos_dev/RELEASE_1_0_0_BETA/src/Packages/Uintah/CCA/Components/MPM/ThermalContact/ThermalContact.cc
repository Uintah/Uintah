#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMPhysicalModules.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <vector>

using namespace Uintah;

ThermalContact::ThermalContact(ProblemSpecP& ps,SimulationStateP& d_sS)
{
  d_sharedState = d_sS;
  lb = scinew MPMLabel();
}

ThermalContact::~ThermalContact()
{
  delete lb;
}

void ThermalContact::computeHeatExchange(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMatls();

  std::vector<NCVariable<double> > gmass(numMatls);
  std::vector<NCVariable<double> > gTemperature(numMatls);
  std::vector<NCVariable<double> > thermalContactHeatExchangeRate(numMatls);

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);
  
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->get(gmass[dwindex], lb->gMassLabel,dwindex, patch, Ghost::None, 0);
    new_dw->get(gTemperature[dwindex], lb->gTemperatureLabel, dwindex, patch,
         Ghost::None, 0);
    new_dw->allocate(thermalContactHeatExchangeRate[dwindex],
         lb->gThermalContactHeatExchangeRateLabel, dwindex, patch);
    thermalContactHeatExchangeRate[dwindex].initialize(0);
  }

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
    double numerator=0.0;
    double denominator=0.0;
    for(int m = 0; m < numMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int n = mpm_matl->getDWIndex();
      double Cp=mpm_matl->getSpecificHeat();
      //double K=mpm_matl->getThermalConductivity();
      numerator   += gTemperature[n][*iter] * gmass[n][*iter] * Cp;
      denominator += gmass[n][*iter] * Cp;
    }
      
    if( !compare(denominator,0.0) ) {
      double contactTemperature = numerator/denominator;
      for(int m = 0; m < numMatls; m++) {
         //double Cp=mpm_matl->getSpecificHeat();
         //double K=mpm_matl->getThermalConductivity();
          if( !compare(gmass[m][*iter],0.0)){
            thermalContactHeatExchangeRate[m][*iter] =
		(contactTemperature - gTemperature[m][*iter])/delT;
          }
          else {
              thermalContactHeatExchangeRate[m][*iter] = 0.0;
          }
      }
    }
  }

  for(int n = 0; n < numMatls; n++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->put(thermalContactHeatExchangeRate[n], 
      lb->gThermalContactHeatExchangeRateLabel, dwindex, patch);
  }

}

void ThermalContact::initializeThermalContact(const Patch* /*patch*/,
					int /*vfindex*/,
					DataWarehouseP& /*new_dw*/)
{
}

void ThermalContact::addComputesAndRequires(Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  t->requires( new_dw, lb->gMassLabel, idx, patch, Ghost::None);
  t->requires( new_dw, lb->gTemperatureLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gThermalContactHeatExchangeRateLabel, idx, patch );
}




