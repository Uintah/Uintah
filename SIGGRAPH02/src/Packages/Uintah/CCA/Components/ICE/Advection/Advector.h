#ifndef UINTAH_ADVECTOR_H
#define UINTAH_ADVECTOR_H

#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

  class DataWarehouse;
  class Patch;

  class Advector {

  public:
    Advector();
    virtual ~Advector();

    virtual Advector* clone(DataWarehouse* new_dw, const Patch* patch) = 0;


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_FC,
                                 const SFCYVariable<double>& vvel_FC,
                                 const SFCZVariable<double>& wvel_FC,
                                 const double& delT, 
                                 const Patch* patch) = 0;


    virtual void advectQ(const CCVariable<double>& q_CC,
                      const Patch* patch,
                      CCVariable<double>& q_advected) = 0;

    virtual void advectQ(const CCVariable<Vector>& q_CC,
                      const Patch* patch,
                      CCVariable<Vector>& q_advected) = 0;

  };

}

#endif
