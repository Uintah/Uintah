#ifndef _TURBULENCE_H
#define _TURBULENCE_H

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

  class DataWarehouse;
  class Patch;
  

  class Turbulence {

  public:
    Turbulence();
    virtual ~Turbulence(); 

    virtual void computeTurbViscosity(DataWarehouse* new_dw,
                                      const Patch* patch,
                                      const CCVariable<Vector>& vel_CC,
                                      const SFCXVariable<double>& uvel_FC,
                                      const SFCYVariable<double>& vvel_FC,
                                      const SFCZVariable<double>& wvel_FC,
                                      const CCVariable<double>& rho_CC,
                                      const int indx,
                                      SimulationStateP&  d_sharedState,
                                      CCVariable<double>& turb_viscosity) = 0;
    
    double d_filter_width;
    
    
  };// End class Turbulence

}// End namespace Uintah

#endif
