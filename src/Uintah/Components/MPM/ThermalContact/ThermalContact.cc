#include <Uintah/Components/MPM/ThermalContact/ThermalContact.h>
#include <Uintah/Components/MPM/Contact/Contact.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NodeIterator.h>

#include <vector>

using namespace Uintah::MPM;

ThermalContact::ThermalContact()
{
}

void ThermalContact::computeHeatExchange(const ProcessorContext*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
 {
  int numMatls = d_sharedState->getNumMatls();
  int NVFs = d_sharedState->getNumVelFields();

  std::vector<NCVariable<double> > gmass(NVFs);
  std::vector<NCVariable<double> > gTemperature(NVFs);
  
  //
  // tan: gExternalHeatRate should be initialized somewhere before.
  //
  std::vector<NCVariable<double> > gExternalHeatRate(NVFs);

  const MPMLabel *lb = MPMLabel::getLabels();
  
  int m, n, vfindex;
  Material* matl;
  MPMMaterial* mpm_matl;
  
  for(m = 0; m < numMatls; m++){
    matl = d_sharedState->getMaterial( m );
    mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], lb->gMassLabel,vfindex , patch,
		  Ghost::None, 0);
      new_dw->get(gTemperature[vfindex], lb->gTemperatureLabel, vfindex, patch,
		  Ghost::None, 0);
      new_dw->get(gExternalHeatRate[vfindex], lb->gExternalHeatRateLabel, 
                  vfindex, patch, Ghost::None, 0);
    }
  }

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++)
  {
    for(n = 0; n < NVFs; n++)
    {
      double temp = gTemperature[n][*iter];
      
      for(m = 0; m < numMatls; m++)
      {
        matl = d_sharedState->getMaterial( m );
        mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        if(mpm_matl){
          vfindex = matl->getVFIndex();
          if( !compare(gTemperature[vfindex][*iter],temp) )
          {
            gExternalHeatRate[n][*iter] += 
              mpm_matl->getHeatTransferCoefficient() *
                ( gmass[vfindex][*iter] * gTemperature[vfindex][*iter] 
                - gmass[n][*iter] * temp );
          }
        }
      }
    }
  }

  for(int n=0; n< NVFs; n++){
    new_dw->put(gExternalHeatRate[n], lb->gExternalHeatRateLabel, n, patch);
  }

}

void ThermalContact::initializeThermalContact(const Patch* patch,
					int vfindex,
					DataWarehouseP& new_dw)
{
}

void ThermalContact::addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{
}


//
// $Log$
// Revision 1.2  2000/05/31 20:51:52  tan
// Finished the computeHeatExchange() function to computer heat exchange.
//
// Revision 1.1  2000/05/31 18:17:27  tan
// Create ThermalContact class to handle heat exchange in
// contact mechanics.
//
//


