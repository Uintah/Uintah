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
    StaticArray<constNCVariable<double> > gTemperature(numMatls);
    StaticArray<NCVariable<double> > thermalContactHeatExchangeRate(numMatls);
#ifdef FRACTURE
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > GTemperature(numMatls);
    StaticArray<NCVariable<double> > GthermalContactHeatExchangeRate(numMatls);
#endif
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);
  
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      new_dw->get(gmass[dwindex], lb->gMassLabel,dwindex, patch, Ghost::None,0);
      new_dw->get(gTemperature[dwindex], lb->gTemperatureLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->allocateTemporary(thermalContactHeatExchangeRate[dwindex],  patch);
      thermalContactHeatExchangeRate[dwindex].initialize(0);
#ifdef FRACTURE
      new_dw->get(Gmass[dwindex],lb->GMassLabel,dwindex, patch, Ghost::None,0);
      new_dw->get(GTemperature[dwindex],lb->GTemperatureLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->allocateTemporary(GthermalContactHeatExchangeRate[dwindex],
				patch);
      GthermalContactHeatExchangeRate[dwindex].initialize(0);
#endif
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      double numerator=0.0;
      double denominator=0.0;
      for(int m = 0; m < numMatls; m++) {
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int n = mpm_matl->getDWIndex();
        double Cp=mpm_matl->getSpecificHeat();
      //double K=mpm_matl->getThermalConductivity();
#ifdef FRACTURE
        numerator   += (gTemperature[n][*iter] * gmass[n][*iter] * Cp
	                 +GTemperature[n][*iter] * Gmass[n][*iter] * Cp); 
        denominator += (gmass[n][*iter] * Cp + Gmass[n][*iter] * Cp);  
#else
        numerator   += gTemperature[n][*iter] * gmass[n][*iter] * Cp;
        denominator += gmass[n][*iter] * Cp;
#endif
      }
      
//      if( !compare(denominator,0.0) ) {
        double contactTemperature = numerator/denominator;
        for(int m = 0; m < numMatls; m++) {
         //double Cp=mpm_matl->getSpecificHeat();
         //double K=mpm_matl->getThermalConductivity();
//          if( !compare(gmass[m][*iter],0.0)){
		thermalContactHeatExchangeRate[m][*iter] =
		(contactTemperature - gTemperature[m][*iter])/delT;
#ifdef FRACTURE
	    GthermalContactHeatExchangeRate[m][*iter] =
	        (contactTemperature - GTemperature[m][*iter])/delT;
#endif
//          }
//          else {
//              thermalContactHeatExchangeRate[m][*iter] = 0.0;
//          }
        }
//      }
    }

    for(int n = 0; n < numMatls; n++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
      int dwindex = mpm_matl->getDWIndex();
      new_dw->put(thermalContactHeatExchangeRate[n], 
        lb->gThermalContactHeatExchangeRateLabel, dwindex, patch);
#ifdef FRACTURE
      new_dw->put(GthermalContactHeatExchangeRate[n],
        lb->GThermalContactHeatExchangeRateLabel, dwindex, patch);
#endif
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
  t->requires(Task::NewDW, lb->gMassLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureLabel, Ghost::None);
  t->computes(lb->gThermalContactHeatExchangeRateLabel);
#ifdef FRACTURE
  t->requires(Task::NewDW, lb->GMassLabel, Ghost::None);
  t->requires(Task::NewDW, lb->GTemperatureLabel, Ghost::None);
  t->computes(lb->GThermalContactHeatExchangeRateLabel);
#endif
}
