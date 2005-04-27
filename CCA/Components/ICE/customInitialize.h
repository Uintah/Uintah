#ifndef _CUSTOMINITIALIZE_H
#define _CUSTOMINITIALIZE_H

#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {

  struct vortices{    // multiple vortices
    vector<Point> origin;
    vector<double> strength;
    vector<double> radius;
    ~vortices() {};
  };
  
  struct mms{         // method of manufactured solutions
    double A;
    ~mms() {};
  };

  struct customInitialize_basket{
    vortices* vortex_inputs;
    mms* mms_inputs;
    string which;
  };
  void customInitialization_problemSetup( const ProblemSpecP& cfd_ice_ps,
                                        customInitialize_basket* cib);
                                        
  void customInitialization(const Patch* patch,
                            CCVariable<double>& rho_CC,
                            CCVariable<double>& temp,
                            CCVariable<Vector>& vel_CC,
                            CCVariable<double>& press_CC,
                            ICEMaterial* ice_matl,
                            const customInitialize_basket* cib);

}// End namespace Uintah
#endif
