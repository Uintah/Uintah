#ifndef _CUSTOMINITIALIZE_H
#define _CUSTOMINITIALIZE_H

#include <Packages/Uintah/Core/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {

  struct customInitialize_basket{
      vector<Point> origin;
      vector<double> strength;
      vector<double> radius;
      string which;
  };
  void customInitialization_problemSetup( const ProblemSpecP& cfd_ice_ps,
                                        customInitialize_basket* cib);
                                        
  void customInitialization(const Patch* patch,
                            CCVariable<double>& rho_CC,
                            CCVariable<double>& temp,
                            CCVariable<Vector>& vel_CC,
                            CCVariable<double>& press_CC,
                            const customInitialize_basket* cib);

}// End namespace Uintah
#endif
