/*
 * SimpleThermalContact.h
 *
 *  Created on: May 3, 2018
 *      Author: jbhooper
 */

#ifndef SRC_CCA_COMPONENTS_MPM_THERMALCONTACT_SIMPLETHERMALCONTACT_H_
#define SRC_CCA_COMPONENTS_MPM_THERMALCONTACT_SIMPLETHERMALCONTACT_H_

#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Components/MPM/Contact/ContactMaterialSpec.h>

#include <CCA/Ports/DataWarehouse.h>

#include <Core/Labels/MPMLabel.h>

namespace Uintah {
  class ProcessorGroup;
  class Patch;
  class VarLabel;
  class Task;

  /**

   CLASS
   SimpleThermalContact

   Implementation of interface bound thermal contact.

   **/

  class SimpleThermalContact: public ThermalContact {
    public:
      SimpleThermalContact( ProblemSpecP      & _inputSpec
                          , SimulationStateP  & _inputManager
                          , MPMLabel          * _inputLabel
                          , MPMFlags          * _inputFlags );

      virtual ~SimpleThermalContact();

      virtual void computeHeatExchange(const  ProcessorGroup  *
                                      ,const  PatchSubset     * patches
                                      ,const  MaterialSubset  * matls
                                      ,       DataWarehouse   * old_dw
                                      ,       DataWarehouse   * new_dw  );

      virtual void initializeThermalContact(const Patch         * patch
                                           ,      int             vfindex
                                           ,      DataWarehouse * new_dw  );

      virtual void addComputesAndRequires(      Task        * task
                                         ,const PatchSet    * patches
                                         ,const MaterialSet * matls   ) const;

      virtual void outputProblemSpec(ProblemSpecP & ps);

    protected:
      SimulationStateP    m_materialManager;
      MPMLabel          * m_mpmLbl;
      MPMFlags          * m_mpmFlags;

      VarLabel          * m_gThermalInterfaceFlag;
      VarLabel          * m_GThermalInterfaceFlag;


      SimpleThermalContact(const SimpleThermalContact & copy);
      SimpleThermalContact& operator=(const SimpleThermalContact &copy);

  };
}



#endif /* SRC_CCA_COMPONENTS_MPM_THERMALCONTACT_SIMPLETHERMALCONTACT_H_ */
