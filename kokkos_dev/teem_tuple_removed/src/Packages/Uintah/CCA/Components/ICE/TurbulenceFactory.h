#ifndef _TURBULENCEFACTORY_H_
#define _TURBULENCEFACTORY_H_

#include <Packages/Uintah/CCA/Components/ICE/Turbulence.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <string>
namespace Uintah {

  class Turbulence;

  class TurbulenceFactory
  {
  public:
    TurbulenceFactory();
    ~TurbulenceFactory();
    
    // this function has a switch for all known turbulence_models
    // and calls the proper class' readParameters()
    
    static Turbulence* create(ProblemSpecP& ps,
                              bool& d_Turb);
                              
    void callTurb(DataWarehouse* new_dw,
                 const Patch* patch,
                 const CCVariable<Vector>& vel_CC,
                 const CCVariable<double>& rho_CC,
                 const int indx,
                 ICELabel* lb,
                 SimulationStateP&  d_sharedState,
                 Turbulence* d_turbulence,
                 CCVariable<double>& tot_viscosity);
  };

} // End namespace Uintah

#endif /*_TURBULENCEFACTORY_H_*/
