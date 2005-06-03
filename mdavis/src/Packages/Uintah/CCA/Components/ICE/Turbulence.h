#ifndef _TURBULENCE_H
#define _TURBULENCE_H

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

  class DataWarehouse;
  class ICELabel;
  class Material;
  class Patch;
  

  class Turbulence {

  public:
    Turbulence();
    Turbulence(ProblemSpecP& ps, SimulationStateP& sharedState);
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

    virtual void scheduleTurbulence1(SchedulerP& sched, const PatchSet* patches,
                                     const MaterialSet* matls) = 0;
   
    void callTurb(DataWarehouse* new_dw,
                 const Patch* patch,
                 const CCVariable<Vector>& vel_CC,
                 const CCVariable<double>& rho_CC,
                 const int indx,
                 ICELabel* lb,
                 SimulationStateP&  d_sharedState,
                 CCVariable<double>& tot_viscosity);
  protected:

    SimulationStateP d_sharedState;
    double d_filter_width;
    
    
    struct FilterScalar {
      string name;
      double scale;
      const VarLabel* scalar;
      const VarLabel* scalarVariance;
      Material* matl;
      MaterialSet* matl_set;
    };
    vector<FilterScalar*> filterScalars;
    
  };// End class Turbulence

}// End namespace Uintah

#endif
