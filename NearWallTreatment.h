#ifndef _NEARWALLTREATMENT_H
#define _NEARWALLTREATMENT_H

#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {

  class NearWallTreatment{

  public:
    NearWallTreatment();
    ~NearWallTreatment(); 

    void computeNearBoundaryWallValue(const Patch* patch,
                                      const CCVariable<Vector>& vel_CC,
                                      const CCVariable<double>& rho_CC,
                                      const int mat_id,
                                      const double viscosity,
                                      CCVariable<double>& turb_viscosity);                             

    void computeNearSolidInterfaceValue(const Patch* patch,
                                        const CCVariable<Vector>& vel_CC,
                                        const CCVariable<double>& rho_CC,
                                        const CCVariable<double>& vol_frac,
                                        const NCVariable<double>& NC_CCweight,
                                        const NCVariable<double>& NCsolidMass,
                                        const NCVariable<Vector>& NCvelocity,
                                        const double viscosity,
                                        CCVariable<double>& turb_viscosity);    
    
  private:
      
  
  };// End class NearWallTreatment

}// End namespace Uintah

#endif
