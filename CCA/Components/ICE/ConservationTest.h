#ifndef _CONSERVATIONTEST_H
#define _CONSERVATIONTEST_H

#include <Packages/Uintah/Core/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {
  void  conservationTest(const Patch* patch,
                         const double& delT,
                         CCVariable<double>& mass_q_CC,
                         constSFCXVariable<double>& uvel_FC,
                         constSFCYVariable<double>& vvel_FC,
                         constSFCZVariable<double>& wvel_FC,
                         double& sum);
}// End namespace Uintah
#endif
 
