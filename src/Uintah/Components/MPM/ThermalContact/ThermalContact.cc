#include <Uintah/Components/MPM/ThermalContact/ThermalContact.h>
#include <Uintah/Components/MPM/Contact/Contact.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Components/MPM/MPMPhysicalModules.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/Task.h>

#include <vector>

using namespace Uintah::MPM;

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
  int NVFs = d_sharedState->getNumVelFields();

  std::vector<NCVariable<double> > gmass(NVFs);
  std::vector<NCVariable<double> > gTemperature(NVFs);
  std::vector<NCVariable<double> > thermalContactHeatExchangeRate(NVFs);

  //  const MPMLabel *lb = MPMLabel::getLabels();
  
  //  int m, n, vfindex;
  Material* matl;
  MPMMaterial* mpm_matl;
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);
  
  for(int m = 0; m < numMatls; m++){
    matl = d_sharedState->getMaterial( m );
    mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], lb->gMassLabel,vfindex , patch,
         Ghost::None, 0);
      new_dw->get(gTemperature[vfindex], lb->gTemperatureLabel, vfindex, patch,
         Ghost::None, 0);
      new_dw->allocate(thermalContactHeatExchangeRate[vfindex],
         lb->gThermalContactHeatExchangeRateLabel, vfindex, patch);
      thermalContactHeatExchangeRate[vfindex].initialize(0);
    }
  }

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
    double numerator=0.0;
    double denominator=0.0;
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(mpm_matl){
        int n = matl->getVFIndex();
        double Cp=mpm_matl->getSpecificHeat();
        //double K=mpm_matl->getThermalConductivity();
        numerator   += gTemperature[n][*iter] * gmass[n][*iter] * Cp;
        denominator += gmass[n][*iter] * Cp;
      }
    }
      
    if( !compare(denominator,0.0) ) {
      double contactTemperature = numerator/denominator;
      for(int m = 0; m < numMatls; m++) {
        matl = d_sharedState->getMaterial( m );
        mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        
        if(mpm_matl){
	    //double Cp=mpm_matl->getSpecificHeat();
          //double K=mpm_matl->getThermalConductivity();
          int n = matl->getVFIndex();
	  if( !compare(gmass[n][*iter],0.0)){
            thermalContactHeatExchangeRate[n][*iter] =
		(contactTemperature - gTemperature[n][*iter])/delT;
          }
	  else {
	    thermalContactHeatExchangeRate[n][*iter] = 0.0;
	  }
        }
      }
    }
  }

  for(int n=0; n< NVFs; n++){
    new_dw->put(thermalContactHeatExchangeRate[n], 
      lb->gThermalContactHeatExchangeRateLabel, n, patch);
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
  //  const MPMLabel* lb = MPMLabel::getLabels();
  t->requires( new_dw, lb->gMassLabel, idx, patch, Ghost::None);
  t->requires( new_dw, lb->gTemperatureLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gThermalContactHeatExchangeRateLabel, idx, patch );
}


//
// $Log$
// Revision 1.14  2000/09/25 20:23:22  sparker
// Quiet g++ warnings
//
// Revision 1.13  2000/08/09 03:18:03  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.12  2000/07/20 19:43:28  guilkey
// Changed the functionality of the ThermalContact to be similar to the
// SingleVelocityField momentum contact algorithm.
//
// Revision 1.11  2000/07/05 23:43:38  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.10  2000/06/28 01:09:39  tan
// Thermal contact model start to work!
//
// Revision 1.9  2000/06/26 18:42:19  tan
// Different heat_conduction properties for different materials are allowed
// in the MPM simulation.
//
// Revision 1.8  2000/06/22 22:33:29  tan
// Moved heat conduction physical parameters (thermalConductivity, specificHeat,
// and heatTransferCoefficient) from MPMMaterial class to HeatConduction class.
//
// Revision 1.7  2000/06/20 05:10:02  tan
// Currently thermal_conductivity, specific_heat and heat_transfer_coefficient
// are set in MPM::MPMMaterial class.
//
// Revision 1.6  2000/06/20 03:41:08  tan
// Get thermal_conductivity, specific_heat and heat_transfer_coefficient
// from ProblemSpecification input requires.
//
// Revision 1.5  2000/06/17 07:06:41  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.4  2000/06/03 05:22:37  sparker
// Adding .cvsignore
//
// Revision 1.3  2000/05/31 22:29:33  tan
// Finished addComputesAndRequires function.
//
// Revision 1.2  2000/05/31 20:51:52  tan
// Finished the computeHeatExchange() function to computer heat exchange.
//
// Revision 1.1  2000/05/31 18:17:27  tan
// Create ThermalContact class to handle heat exchange in
// contact mechanics.
//
//


